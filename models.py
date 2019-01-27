import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=False):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.activation = nn.Tanh()
        layer = []
        for i in range(len(hidden_dim)):
            if i == 0:
                layer.append(nn.Linear(input_dim + hidden_dim[0], hidden_dim[0], bias=self.bias))
            else:
                layer.append(nn.Linear(hidden_dim[i-1], hidden_dim[i], bias=self.bias))
            layer.append(self.activation)
        self.main = nn.Sequential(*layer)
        self.weight = nn.Linear(self.hidden_dim[-1], 4 * hidden_dim[-1], bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        x = self.main(combined) 
        x = self.weight(x)
        x = torch.tanh(x)
        cc_i, cc_f, cc_o, cc_g = torch.split(x, self.hidden_dim[-1], dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class LSTM(nn.Module):
    def __init__(self, input_dim=8, lstm_hidden_dim=[30, 45, 40, 30], time_step=3):  # 3 + 5
        super(LSTM, self).__init__()
        self.time_step = time_step
        # for i in range(self.time_step):
            # self.lstm.append(LSTMCell(input_dim, lstm_hidden_dim, bias=True))
        self.hidden_transform = nn.Linear(input_dim + lstm_hidden_dim[0], lstm_hidden_dim[0])
        self.lstm = LSTMCell(lstm_hidden_dim[0], lstm_hidden_dim, bias=True)
        self.predict1 = nn.Linear(lstm_hidden_dim[-1], 10)
        self.predict2 = nn.Linear(10, 1)

    def forward(self, x):
        batch_size = x.size()[0]
        hidden = torch.zeros(batch_size, 30).cuda()
        cell = torch.zeros(batch_size, 30).cuda()
        for i in range(self.time_step):
            hidden_state = torch.cat([x[:, i, :], hidden], dim=1)
            hidden_state = self.hidden_transform(hidden_state)
            hidden, cell = self.lstm(hidden_state, [hidden_state, cell])
        x = self.predict1(hidden)
        x = self.predict2(x)
                
        return x


if __name__ == "__main__":
    model = LSTM(input_dim=8, lstm_hidden_dim=[30, 45, 40, 30], time_step=3).cuda()
    print(model)
    inputs = torch.randn(6, 3, 8).cuda()
    output = model(inputs)
    print(output.shape)
    print(output.mean())
