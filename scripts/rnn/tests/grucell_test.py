import unittest
import torch
import torch.nn as nn
from torch.nn.modules import rnn
from gru import MyGRUCell


class TestMyGRUCell(unittest.TestCase):
    def test_comp(self):
        seq_len = 1
        input_size = 2
        for batch in range(1, 6):
            inputs = torch.rand(seq_len, batch, input_size)
            torch.manual_seed(0)
            rnn_cell = nn.GRUCell(input_size=input_size, hidden_size=2)
            torch.manual_seed(0)
            my_cell = MyGRUCell(input_size=input_size, hidden_size=2)

            h0 = torch.rand(batch, 2)

            hx = h0.detach().clone()
            output1 = []
            for i in range(seq_len):
                hx = rnn_cell(inputs[i], hx)
                output1.append(hx)

            hx = h0.detach().clone()
            output2 = []
            for i in range(seq_len):
                hx = my_cell(inputs[i], hx)
                output2.append(hx)

            self.assertEqual(len(output1), len(output2))
            for i in range(len(output1)):
                self.assertTrue(torch.allclose(output1[i], output2[i]))

    def test_shape(self):
        sel_len = 30
        batch = 1
        input_size = 10
        hidden_size = 64

        inputs = torch.rand(sel_len, batch, input_size)
        torch.manual_seed(5)
        rnn_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        torch.manual_seed(5)
        my_cell = MyGRUCell(input_size=input_size, hidden_size=hidden_size)

        h0 = torch.rand(batch, hidden_size)

        hx1 = h0.detach().clone()
        hx1 = rnn_cell(inputs[0], hx1)

        hx2 = h0.detach().clone()
        hx2 = my_cell(inputs[0], hx2)

        self.assertEqual(hx1.shape, hx2.shape)
        self.assertEqual(hx1.shape, (batch, hidden_size))
