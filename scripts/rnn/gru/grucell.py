import math
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.tensor import Tensor
import torch.nn.functional as F


class MyGRUCell(nn.Module):
    input_size: int
    hidden_size: int
    chunk_size: int
    bias: bool
    w_ih: Tensor
    w_hh: Tensor

    def __init__(
        self, input_size: int, hidden_size: int, chunk_size=3, bias: bool = True
    ) -> None:
        super(MyGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.bias = bias

        self.w_ih = Parameter(Tensor(chunk_size * hidden_size, input_size))
        self.w_hh = Parameter(Tensor(chunk_size * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(Tensor(chunk_size * hidden_size))
            self.bias_hh = Parameter(Tensor(chunk_size * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(
                x.size(0), self.hidden_size, dtype=x.dtype, device=x.device
            )
        x = x.view(-1, x.size(1))
        if self.bias:
            gate_x = torch.matmul(x, torch.transpose(self.w_ih, 0, 1)) + self.bias_ih
            gate_h = torch.matmul(hx, torch.transpose(self.w_hh, 0, 1)) + self.bias_hh
        else:
            gate_x = torch.matmul(x, torch.transpose(self.w_ih, 0, 1))
            gate_h = torch.matmul(hx, torch.transpose(self.w_hh, 0, 1))
        i_r, i_i, i_n = gate_x.chunk(self.chunk_size, 1)
        h_r, h_i, h_n = gate_h.chunk(self.chunk_size, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hx - newgate)
        return hy
