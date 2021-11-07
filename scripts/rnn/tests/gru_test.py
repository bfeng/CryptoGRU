import unittest
import torch
import torch.nn as nn
from gru import MyGRU, MyGRUCell


class TestMyGRU(unittest.TestCase):
    def test_comp(self):
        seq_len = 1
        input_size = 2
        batch = 1

        inputs = torch.rand(seq_len, batch, input_size)
        h0 = torch.rand(1, batch, 2)
        torch.manual_seed(0)
        gru = nn.GRU(input_size=input_size, hidden_size=2, num_layers=1)
        output1, hn1 = gru(inputs, h0)

        torch.manual_seed(0)
        my_gru = MyGRU(input_size=input_size, hidden_size=2, num_layers=1)
        # my_gru = nn.GRU(input_size=input_size, hidden_size=2, num_layers=1)
        output2, hn2 = my_gru(inputs, h0)
        self.assertEqual(hn1.shape, hn2.shape)
        self.assertEqual(output1.shape, output2.shape)
        self.assertTrue(torch.allclose(hn1, hn2))
        self.assertTrue(torch.allclose(output1, output2))

    def test_comp_gru(self):
        seq_len = 1
        input_size = 2
        batch = 1
        inputs = torch.rand(seq_len, batch, input_size)
        h0 = torch.zeros(1, batch, 2)
        torch.manual_seed(0)
        gru = nn.GRU(input_size=input_size, hidden_size=2, num_layers=1)
        output1, hn1 = gru(inputs, h0)

        torch.manual_seed(0)
        my_gru_cell = MyGRUCell(input_size=input_size, hidden_size=2)
        output2 = []
        hx = torch.zeros(batch, 2)
        for i in range(seq_len):
            hx = my_gru_cell(inputs[i], hx)
            output2.append(hx)
        output2 = torch.stack(output2)
        self.assertTrue(torch.allclose(hn1, hx))
