from gru import MyGRUCell
import torch
import torch.nn as nn


if __name__ == "__main__":
    torch.manual_seed(0)

    seq_len = 10
    input_size = 2
    batch = 1

    inputs = torch.randn(seq_len, batch, input_size)
    print(inputs)
    h0 = torch.randn(1, batch, 2)
    rnn = nn.GRU(input_size=input_size, hidden_size=2, num_layers=1)
    output1, hn = rnn(inputs, h0)
    print(output1)

    rnn_cell = nn.GRUCell(input_size=input_size, hidden_size=2)

    output2 = []
    hx = h0[0, :, :]
    for i in range(seq_len):
        hx = rnn_cell(inputs[i], hx)
        output2.append(hx)

    output2 = torch.stack(output2)
    print(output2)
    # comp = torch.allclose(hn, output2)
    # print(comp)

    my_gru_cel = MyGRUCell(input_size=input_size, hidden_size=2, bias=False)

    output3 = []
    hx = h0[0, :, :]
    for i in range(seq_len):
        hx = my_gru_cel(inputs[i], hx)
        output3.append(hx)

    output3 = torch.stack(output3)
    print(output3)
    comp = torch.allclose(output2, output3)
    print(comp)
