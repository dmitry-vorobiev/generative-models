import torch
from torch import Tensor
from typing import Dict, Optional, Callable, Mapping, Tuple, Union

Batch = Union[Tensor, Tuple[Tensor, Tensor]]
Device = Optional[torch.device]

FloatDict = Dict[str, float]
TensorMap = Mapping[str, Tensor]

G = torch.nn.Module
D = torch.nn.Module

GLossFunc = Callable[[G, D, Tensor, Optional[Tensor], Optional[FloatDict]], Tensor]
DLossFunc = Callable[[G, D, Tensor, Tensor, Optional[Tensor], Optional[FloatDict]], Tensor]

TrainFunc = Callable[[int, Batch], FloatDict]
SnapshotFunc = Callable[[], Tensor]
