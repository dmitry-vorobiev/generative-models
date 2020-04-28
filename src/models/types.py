import torch
from torch import Tensor
from torch.nn import Module
from typing import Dict, Optional, Callable, Mapping, Tuple, Union

Batch = Union[Tensor, Tuple[Tensor, Tensor]]
Device = Optional[torch.device]

FloatDict = Dict[str, float]
TensorMap = Mapping[str, Tensor]

GLossFunc = Callable[[Module, Module, Tensor, Optional[Tensor]], Tensor]
DLossFunc = Callable[[Module, Module, Tensor, Tensor, Optional[Tensor]], Tensor]

TrainFunc = Callable[[Batch], FloatDict]
