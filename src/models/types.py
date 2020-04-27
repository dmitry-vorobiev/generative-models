import torch
from torch import Tensor
from torch.nn import Module
from typing import Optional, Callable, Mapping, Tuple, Union

Batch = Tuple[Tensor, Optional[Tensor]]
Device = Optional[torch.device]

TensorMap = Mapping[str, Tensor]

GLossFunc = Callable[[Module, Module, Tensor, Optional[Tensor]], Tensor]
DLossFunc = Callable[[Module, Module, Tensor, Tensor, Optional[Tensor]], Tensor]

TrainFunc = Callable[[Tensor, Optional[Tensor]], TensorMap]
