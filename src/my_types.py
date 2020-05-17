import torch
from torch import Tensor
from typing import Dict, Optional, Callable, Mapping, Tuple, Union

Batch = Union[Tensor, Tuple[Tensor, Tensor]]
Device = Optional[torch.device]

FloatDict = Dict[str, float]
TensorMap = Mapping[str, Tensor]

G = torch.nn.Module
D = torch.nn.Module

LossWithStats = Tuple[Tensor, FloatDict]
GLossFunc = Callable[[G, D, Tensor, Optional[Tensor], Optional[FloatDict]], LossWithStats]
DLossFunc = Callable[[G, D, Tensor, Tensor, Optional[Tensor], Optional[FloatDict]], LossWithStats]


TrainFunc = Callable[[int, Batch], FloatDict]
SampleImages = Callable[[], Tensor]
