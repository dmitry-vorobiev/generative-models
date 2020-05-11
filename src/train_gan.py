import datetime as dt
import hydra
import logging
import os
import time
import torch
import torch.distributed as dist
import torchvision

from hydra.utils import instantiate
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Metric, RunningAverage
from ignite.utils import convert_tensor
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from typing import Any, Dict, List, Optional, Tuple, Sized

from data.dataset import JustImages
from models.stylegan2.train import create_train_closures
from my_types import Batch, Device, FloatDict, SnapshotFunc, TrainFunc

Metrics = Dict[str, Metric]


def humanize_time(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')


def on_epoch_start(engine: Engine):
    engine.state.t0 = time.time()


def log_iter(engine: Engine, trainer: Engine, pbar: ProgressBar, log_interval: int) -> None:
    epoch = trainer.state.epoch
    iteration = engine.state.iteration
    metrics = engine.state.metrics
    stats = {k: '%.3f' % v for k, v in metrics.items()}
    stats = ', '.join(['{}: {}'.format(*e) for e in stats.items()])
    t0 = engine.state.t0
    t1 = time.time()
    it_time = (t1 - t0) / log_interval
    cur_time = humanize_time(t1)
    pbar.log_message("[{}][{:.2f} s] | ep: {:2d}, it: {:3d}, {}".format(
        cur_time, it_time, epoch, iteration, stats))
    engine.state.t0 = t1


def log_epoch(engine: Engine) -> None:
    epoch = engine.state.epoch
    metrics = engine.state.metrics
    stats = {k: '%.3f' % v for k, v in metrics.items()}
    stats = ', '.join(['{}: {}'.format(*e) for e in stats.items()])
    logging.info("ep: {}, {}".format(epoch, stats))


def create_trainer(train_func, metrics=None, device=None):
    # type: (TrainFunc, Optional[Metrics], Device) -> Engine

    def _update(e: Engine, batch: Batch) -> FloatDict:
        iteration = e.state.iteration - 1  # it starts from 1
        loss = train_func(iteration, *batch)
        return loss

    trainer = Engine(_update)
    if metrics:
        for name, metric in metrics.items():
            metric.attach(trainer, name)
    return trainer


def _prepare_batch(batch: Batch, device: torch.device, non_blocking: bool) -> Tuple[Any, Any]:
    if isinstance(batch, tuple):
        x, y = batch
        return (convert_tensor(x, device=device, non_blocking=non_blocking),
                convert_tensor(y, device=device, non_blocking=non_blocking))
    else:
        return convert_tensor(batch, device=device, non_blocking=non_blocking)


def create_metrics(keys: List[str], device: Device = None) -> Metrics:
    def _out_transform(kek: str):
        return lambda out: out[kek]

    metrics = {key: RunningAverage(output_transform=_out_transform(key),
                                   device=device)
               for key in keys}
    return metrics


def _upd_pbar_iter_from_cp(engine: Engine, pbar: ProgressBar) -> None:
    pbar.n = engine.state.iteration


def create_simple_dataset(conf, transforms):
    # type: (DictConfig, DictConfig) -> JustImages
    transforms = T.Compose([instantiate(v) for k, v in transforms.items()])
    ds = JustImages(conf.root, extensions=tuple(conf.extensions), transform=transforms)
    return ds


def create_image_folder_dataset(conf, transforms):
    # type: (DictConfig, DictConfig) -> ImageFolder
    transforms = T.Compose([instantiate(v) for k, v in transforms.items()])
    return ImageFolder(conf.root, transform=transforms)


def default_collate_no_labels(batch):
    # type: (List[Tensor]) -> Tuple[Tensor, None]
    """
    Copy-paste from torch/utils/data/_utils/collate.py default_collate
    """
    elem = batch[0]
    assert isinstance(elem, torch.Tensor)
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    return torch.stack(batch, 0, out=out), None


def create_train_loader(conf, rank=None, num_replicas=None):
    # type: (DictConfig, Optional[int], Optional[int]) -> Sized
    build_ds = {
        'simple': create_simple_dataset,
        'image_folder': create_image_folder_dataset,
    }
    collate_funcs = {
        'simple': default_collate_no_labels,
        'image_folder': None,
    }
    ds_type = conf.type
    data = build_ds[ds_type](conf, conf.transforms)
    print("Found {} images".format(len(data)))

    sampler = None
    if num_replicas is not None:
        sampler = DistributedSampler(data, num_replicas=num_replicas, rank=rank)

    loader = DataLoader(data,
                        sampler=sampler,
                        collate_fn=collate_funcs[ds_type],
                        batch_size=conf.loader.batch_size,
                        num_workers=conf.get('loader.workers', 0),
                        drop_last=True)
    return loader


def handle_snapshot_images(engine: Engine, make_snapshot: SnapshotFunc, save_dir: str):
    images = make_snapshot()
    path = os.path.join(save_dir, '%06d.png' % engine.state.iteration)
    torchvision.utils.save_image(images, path)


def setup_snapshots(trainer: Engine, make_snapshot: SnapshotFunc, conf: DictConfig):
    snapshots = conf.snapshots
    use_ema = conf.train.G_ema
    if snapshots.enabled:
        if use_ema:
            snap_event = Events.ITERATION_COMPLETED(every=snapshots.interval_iteration)
            snap_path = snapshots.get('save_dir', os.path.join(os.getcwd(), 'images'))
            if not os.path.exists(snap_path):
                os.makedirs(snap_path)
            logging.info("Saving snapshot images to {}".format(snap_path))
            trainer.add_event_handler(snap_event, handle_snapshot_images, make_snapshot, snap_path)
        else:
            logging.warning("Snapshot generation requires train.G_ema=true. "
                            "Snapshots will be turned off for this run.")


def setup_checkpoints(trainer, obj_to_save, epoch_length, conf):
    # type: (Engine, Dict[str, Any], int, DictConfig) -> None
    cp = conf.checkpoints
    save_path = cp.get('save_dir', os.getcwd())
    logging.info("Saving checkpoints to {}".format(save_path))
    max_cp = max(int(cp.get('max_checkpoints', 1)), 1)
    save = DiskSaver(save_path, create_dir=True, require_empty=True)
    make_checkpoint = Checkpoint(obj_to_save, save, n_saved=max_cp)
    cp_iter = cp.interval_iteration
    cp_epoch = cp.interval_epoch
    if cp_iter > 0:
        save_event = Events.ITERATION_COMPLETED(every=cp_iter)
        trainer.add_event_handler(save_event, make_checkpoint)
    if cp_epoch > 0:
        if cp_iter < 1 or epoch_length % cp_iter:
            save_event = Events.EPOCH_COMPLETED(every=cp_epoch)
            trainer.add_event_handler(save_event, make_checkpoint)


def run(conf: DictConfig, local_rank=0):
    epochs = conf.train.epochs
    epoch_length = conf.train.epoch_length
    torch.manual_seed(conf.general.seed)

    dist_conf = conf.distributed
    backend = dist_conf.backend
    distributed = backend is not None

    if distributed:
        rank = dist.get_rank()
        num_replicas = dist.get_world_size()
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        num_replicas = 1
        torch.cuda.set_device(conf.general.gpu)
    device = torch.device('cuda')
    loader_args = dict()
    master_node = rank == 0

    if master_node:
        print(conf.pretty())
    if num_replicas > 1:
        epoch_length = epoch_length // num_replicas
        loader_args = dict(rank=rank, num_replicas=num_replicas)

    train_dl = create_train_loader(conf.data, **loader_args)

    if epoch_length < 1:
        epoch_length = len(train_dl)

    metric_names = list(conf.logging.stats)
    metrics = create_metrics(metric_names, device if distributed else None)

    G = instantiate(conf.model.G).to(device)
    D = instantiate(conf.model.D).to(device)
    G_loss = instantiate(conf.loss.G).to(device)
    D_loss = instantiate(conf.loss.D).to(device)
    G_opt = instantiate(conf.optim.G, G.parameters())
    D_opt = instantiate(conf.optim.D, D.parameters())
    G_ema = None

    if conf.train.G_ema and master_node:
        G_ema = instantiate(conf.model.G)
        if not conf.train.G_ema_on_cpu:
            G_ema = G_ema.to(device)
        G_ema.load_state_dict(G.state_dict())
        G_ema.requires_grad_(False)

    to_save = {
        'G': G,
        'D': D,
        'G_loss': G_loss,
        'D_loss': D_loss,
        'G_opt': G_opt,
        'D_opt': D_opt,
        'G_ema': G_ema
    }

    if master_node and conf.logging.model:
        logging.info(G)
        logging.info(D)

    if distributed:
        ddp_kwargs = dict(device_ids=[local_rank, ], output_device=local_rank)
        G = torch.nn.parallel.DistributedDataParallel(G, **ddp_kwargs)
        D = torch.nn.parallel.DistributedDataParallel(D, **ddp_kwargs)

    train_options = {
        'train':    dict(conf.train),
        'snapshot': dict(conf.snapshots)
    }
    bs_dl = int(conf.data.loader.batch_size) * num_replicas
    bs_eff = conf.train.batch_size
    if bs_eff % bs_dl:
        raise AttributeError("Effective batch size should be divisible by data-loader batch size "
                             "multiplied by number of devices in use")  # until there is no special bs for master node...
    upd_interval = max(bs_eff // bs_dl, 1)
    train_options['train']['update_interval'] = upd_interval
    if epoch_length < len(train_dl):
        # ideally epoch_length should be tied to the effective batch_size only
        # and the ignite trainer counts data-loader iterations
        epoch_length *= upd_interval

    train_loop, make_snapshot = create_train_closures(
        G, D, G_loss, D_loss, G_opt, D_opt, G_ema=G_ema, device=device, options=train_options)
    trainer = create_trainer(train_loop, metrics, device)
    to_save['trainer'] = trainer

    every_iteration = Events.ITERATION_COMPLETED
    trainer.add_event_handler(every_iteration, TerminateOnNan())

    cp = conf.checkpoints
    pbar = None

    if master_node:
        log_freq = conf.logging.iter_freq
        log_event = Events.ITERATION_COMPLETED(every=log_freq)
        pbar = ProgressBar(persist=False)
        trainer.add_event_handler(Events.EPOCH_STARTED, on_epoch_start)
        trainer.add_event_handler(log_event, log_iter, trainer, pbar, log_freq)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_epoch)
        pbar.attach(trainer, metric_names=metric_names)
        setup_checkpoints(trainer, to_save, epoch_length, conf)
        setup_snapshots(trainer, make_snapshot, conf)

    if 'load' in cp.keys() and cp.load is not None:
        if master_node:
            logging.info("Resume from a checkpoint: {}".format(cp.load))
            trainer.add_event_handler(Events.STARTED, _upd_pbar_iter_from_cp, pbar)
        Checkpoint.load_objects(to_load=to_save,
                                checkpoint=torch.load(cp.load, map_location=device))

    try:
        trainer.run(train_dl, max_epochs=epochs, epoch_length=epoch_length)
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
    if pbar is not None:
        pbar.close()


@hydra.main(config_path="../config/train_gan.yaml")
def main(conf: DictConfig):
    env = os.environ.copy()
    local_rank = int(env.get('LOCAL_RANK', 0))
    dist_conf: DictConfig = conf.distributed
    backend = dist_conf.backend
    distributed = backend is not None

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Unable to find any CUDA device")

        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend, init_method=dist_conf.url)
        if local_rank == 0:
            print("\nDistributed setting:")
            print("\tbackend: {}".format(dist.get_backend()))
            print("\tworld size: {}".format(dist.get_world_size()))
            print("\trank: {}\n".format(dist.get_rank()))

    try:
        run(conf, local_rank)
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        if distributed:
            dist.destroy_process_group()
        raise e

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
