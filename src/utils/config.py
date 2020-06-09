from omegaconf import DictConfig
from typing import List


def read_int_list(conf: DictConfig, key: str) -> List[int]:
    seq = conf[key]
    if seq is None:
        return []
    elif isinstance(seq, int):
        seq = [seq]
    elif isinstance(seq, str):
        interval = list(map(str.strip, seq.split('-')))
        if len(interval) == 2:
            interval = map(int, interval)
            seq = list(range(*interval))
        elif len(interval) < 2:
            seq = list(map(str.strip, seq.split(',')))
        else:
            raise AttributeError("Incorrect format for '{}'".format(key))
    return seq
