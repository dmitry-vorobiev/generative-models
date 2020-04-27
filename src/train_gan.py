import datetime as dt
import hydra
import logging
import os
import time
import torch
import torch.distributed as dist

from hydra.utils import instantiate
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Accuracy, Loss, Metric, RunningAverage
from ignite.utils import convert_tensor
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .models.stylegan2.train import create_train_loop
from .models.types import Batch, Device, TensorMap, TrainFunc

Metrics = Dict[str, Metric]


def humanize_time(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')


def on_epoch_start(engine: Engine):
    engine.state.t0 = time.time()


def log_iter(engine: Engine, trainer: Engine, pbar: ProgressBar, log_interval: int) -> None:
    epoch = trainer.state.epoch
    iteration = engine.state.iteration
    metrics = engine.state.metrics
    stats = {k: '%.3f' % v for k, v in metrics.items() if 'loss' in k}
    if hasattr(engine.state, 'lr'):
        stats['lr'] = ', '.join(['%.1e' % val for val in engine.state.lr])
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
    stats = {k: '%.3f' % metrics[k] for k in ['acc', 'nll']}
    stats = ', '.join(['{}: {}'.format(*e) for e in stats.items()])
    logging.info("ep: {}, {}".format(epoch, stats))


def create_trainer(train_func: TrainFunc, metrics: Optional[Metrics] = None):

    def _update(e: Engine, batch: Batch) -> TensorMap:
        iteration = e.state.iteration
        loss = train_func(batch)
        return loss

    engine = Engine(_update)
    if metrics:
        add_metrics(engine, metrics)
    return engine


def add_metrics(engine: Engine, metrics: Metrics):
    for name, metric in metrics.items():
        metric.attach(engine, name)


def _prepare_batch(batch: Any, device: torch.device,
                   non_blocking: bool) -> Tuple[Any, Any]:
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def create_metrics(keys: List[str], device: Device = None) -> Metrics:
    def _out_transform(key: str):
        return lambda out: out[key]

    metrics = {key: RunningAverage(output_transform=_out_transform(key))
               for key in keys}
    if device:
        for m in metrics.values():
            m.device = device
    return metrics


def _upd_pbar_iter_from_cp(engine: Engine, pbar: ProgressBar) -> None:
    pbar.n = engine.state.iteration


def run(conf: DictConfig):
    epochs = conf.train.epochs
    epoch_length = conf.train.epoch_length
    torch.manual_seed(conf.general.seed)

    dist_conf = conf.distributed
    local_rank = dist_conf.local_rank
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

    if rank == 0:
        print(conf.pretty())

    if num_replicas > 1:
        epoch_length = epoch_length // num_replicas
        loader_args = dict(rank=rank, num_replicas=num_replicas)
    else:
        loader_args = dict()

    train_dl = create_train_loader(conf.data.train, epoch_length=epoch_length, **loader_args)

    if epoch_length < 1:
        epoch_length = len(train_dl)

    model = instantiate(conf.model).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank, ], output_device=local_rank)
        model.to_y = model.module.to_y
    if rank == 0 and conf.logging.model:
        print(model)

    metric_names = ['G_loss', 'D_loss']
    metrics = create_metrics(metric_names, device if distributed else None)

    G = instantiate(conf.model.G)
    D = instantiate(conf.model.D)
    G_loss = instantiate(conf.loss.G)
    D_loss = instantiate(conf.loss.D)
    G_opt = instantiate(conf.optim.G)
    D_opt = instantiate(conf.optim.D)

    train_loop = create_train_loop(G, D, G_loss, D_loss, G_opt, D_opt, 0, device)
    trainer = create_trainer(train_loop, metrics)

    every_iteration = Events.ITERATION_COMPLETED
    trainer.add_event_handler(every_iteration, TerminateOnNan())

    cp = conf.train.checkpoints
    to_save = {
        'trainer': trainer,
        'G': G,
        'D': D,
        'G_loss': G_loss,
        'D_loss': D_loss,
        'G_opt': G_opt,
        'D_opt': D_opt,
    }
    save_path = cp.get('base_dir', os.getcwd())
    pbar = None

    if rank == 0:
        log_freq = conf.logging.iter_freq
        log_event = Events.ITERATION_COMPLETED(every=log_freq)
        pbar = ProgressBar(persist=False)

        trainer.add_event_handler(Events.EPOCH_STARTED, on_epoch_start)
        trainer.add_event_handler(log_event, log_iter, trainer, pbar, log_freq)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_epoch)
        pbar.attach(trainer, metric_names=metric_names)

        if 'load' in cp.keys() and cp.load:
            logging.info("Resume from a checkpoint: {}".format(cp.load))
            trainer.add_event_handler(Events.STARTED, _upd_pbar_iter_from_cp, pbar)

        logging.info("Saving checkpoints to {}".format(save_path))

    if rank == 0:
        max_cp = max(int(cp.get('max_checkpoints', 1)), 1)
        save = DiskSaver(save_path, create_dir=True, require_empty=True)
        make_checkpoint = Checkpoint(to_save, save, n_saved=max_cp)
        cp_iter = cp.interval_iteration
        cp_epoch = cp.interval_epoch
        if cp_iter > 0:
            save_event = Events.ITERATION_COMPLETED(every=cp_iter)
            trainer.add_event_handler(save_event, make_checkpoint)
        if cp_epoch > 0:
            if cp_iter < 1 or epoch_length % cp_iter:
                save_event = Events.EPOCH_COMPLETED(every=cp_epoch)
                trainer.add_event_handler(save_event, make_checkpoint)

    if 'load' in cp.keys() and cp.load:
        Checkpoint.load_objects(to_load=to_save,
                                checkpoint=torch.load(cp.load, map_location=device))

    try:
        trainer.run(train_dl, max_epochs=epochs, epoch_length=epoch_length)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    if pbar is not None:
        pbar.close()


@hydra.main(config_path="../config/train_gan.yaml")
def main(conf: DictConfig):
    dist_conf = conf.distributed
    local_rank = dist_conf.local_rank
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
        run(conf)
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
