from typing import Tuple
import torch
from torch.tensor import Tensor
from .grucell import MyGRUCell
import torch.nn as nn


class MyGRU(nn.Module):
    input_size: int
    hidden_size: int
    num_layers: int
    bias: int

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True
    ) -> None:
        super(MyGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.gru_cell = MyGRUCell(
            input_size=input_size, hidden_size=hidden_size, chunk_size=3, bias=bias
        )

    def reset_parameters(self) -> None:
        self.gru_cell.reset_parameters()

    def forward(self, x: Tensor, h0: Tensor) -> Tuple[Tensor, Tensor]:
        seq_len = x.size(0)
        output = []
        hx = h0[0, :, :]
        for i in range(seq_len):
            hx = self.gru_cell(x[i], hx)
            output.append(hx)

        return torch.stack(output), hx
