import torch
from torch import Tensor
from torch.nn import Module
from typing import Dict, Optional, Callable, Mapping, Tuple, Union

Batch = Tuple[Tensor, Optional[Tensor]]
Device = Optional[torch.device]

FloatDict = Dict[str, float]
TensorMap = Mapping[str, Tensor]

GLossFunc = Callable[[Module, Module, Tensor, Optional[Tensor], Optional[FloatDict]], Tensor]
DLossFunc = Callable[[Module, Module, Tensor, Tensor, Optional[Tensor], Optional[FloatDict]], Tensor]

TrainFunc = Callable[[int, Tensor, Optional[Tensor]], FloatDict]
SnapshotFunc = Callable[[], Tensor]
