import numpy as np
import torch
import torch.nn as nn


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim)
        return hidden


def train(model, train_loader, num_epochs=1):
    criterion = nn.MSELoss()
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    print("Start training of {} model".format("GRU"))
    history = {"loss": []}
    it = 0
    for epoch in range(num_epochs):
        print(f"epoch-{epoch}:")
        h = model.init_hidden(1)
        for inputs, target in train_loader:
            print(f"Iter: {it}, inputs: {inputs.shape}, target: {target}")
            optimizer.zero_grad()
            outputs, h = model(inputs, h)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            history.loss.append(loss.item())
            it += 1
    print("Model trained.")
    return model


def main():
    input_dim = 10
    hidden_dim = 4
    output_dim = 1
    x = [np.arange(i * 10, i * 10 + 10) for i in range(10)]
    x = np.array(x)
    x = x[:, np.newaxis, :, np.newaxis]
    y = [(i + 1) * 10 for i in range(10)]
    train_loader = zip(x, y)
    model = GRUNet(input_dim, hidden_dim, output_dim, 1)
    model = train(model, train_loader)


if __name__ == "__main__":
    main()
