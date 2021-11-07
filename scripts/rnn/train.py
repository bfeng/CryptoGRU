import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class GRUCell(nn.Module):

    input_size: int
    hidden_size: int
    bias: bool

    def __init__(
        self, input_size: int, hidden_size: int, layer_dim: int, bias: bool = True
    ):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy


class GRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_dim: int,
        output_dim: int,
        bias=True,
    ):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim))
        outs = []
        hn = h0[0, :, :]
        for seq in range(x.shape(1)):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data


def train(model, train_loader, num_epochs: int = 1):
    criterion = nn.MSELoss()
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    print("Start training of {} model".format("GRU"))
    history = {"loss": []}
    it = 0
    for epoch in range(num_epochs):
        print(f"epoch-{epoch}:")
        for batch, target in train_loader:
            print(f"Iter: {it}, batch: {batch}, target: {target}")
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            history.loss.append(loss.item())
            it += 1


def main():
    input_dim = 10
    hidden_dim = 4
    output_dim = 1
    x = [np.arange(i * 10, i * 10 + 10) for i in range(10)]
    x = np.array(x)
    x = x[:, :, np.newaxis]
    y = [(i + 1) * 10 for i in range(10)]
    train_loader = zip(x, y)
    model = GRUModel(input_dim, hidden_dim, 1, output_dim)
    train(model, train_loader)


if __name__ == "__main__":
    main()
