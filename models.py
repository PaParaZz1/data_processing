import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=False):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        layer = []
        for i in range(len(hidden_dim)):
            if i == 0:
                layer.append(nn.Linear(input_dim + hidden_dim[0], hidden_dim[0], bias=self.bias))
            else:
                layer.append(nn.Linear(hidden_dim[i-1], hidden_dim[i], bias=self.bias))
        self.main = nn.Sequential(*layer)
        self.fc1 = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=self.bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=self.bias)
        self.W = nn.Linear(hidden_dim, 4 * hidden_dim, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        combined_conv = F.relu(self.W(x))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class LSTM(nn.Module):
    def __init__(self, input_dim=10, lstm_hidden_dim=[40, 30, 15], **kwargs):  # 3 + 7
        super(LSTM, self).__init__()
        self.lstm = LSTMCell(input_dim, lstm_hidden_dim, bias=True)
        self.predict = nn.Linear(lstm_hidden_dim[-1], 1)

    def forward(self, x):
        return 0


class pred_net(nn.Module):
    def __init__(self, s_dim, a_dim, p_dim, frame_history_len=3, args=None):
        super(pred_net, self).__init__()
        self.fc1 = nn.Linear(s_dim * frame_history_len, s_dim * 4)
        self.fc = nn.Linear(s_dim * 4 + a_dim * 4 + p_dim, s_dim * 4)
        self.act_fc = nn.Linear(a_dim, a_dim * 4)
        self.fc2 = nn.Linear(s_dim * 4, s_dim * frame_history_len)
        self.lstm = LSTMCell(s_dim * 4, s_dim * 4, bias=False)
        self.reward_layer = nn.Linear(s_dim * 4 + 10, 1)
        self.s_dim = s_dim
        self.value_fc = nn.Linear(s_dim * frame_history_len, 1)
        self.frame_history_len = frame_history_len
        self.args = args

    def get_value(self, states):
        # states : batch * pred_step * 4s_dim
        batch_size = states.size(0)
        pred_step = states.size(1)
        states = states.view(-1, self.s_dim * self.frame_history_len)
        values = self.value_fc(states)
        return values.view(batch_size, pred_step)

    def forward(self, state, act_sequence, latent):
        batch_size = int(state.size()[0])
        state = F.relu(self.fc1(state))
        hidden_state = torch.cat([state, self.act_fc(act_sequence[:, 0, :]), latent], dim=1)
        hidden = F.relu(self.fc(hidden_state))
        cell = hidden
        pred_step = act_sequence.size(1)
        rewards, preds, values = [], [], []
        steps = Variable(torch.ones((batch_size, 10))).to('cuda')
        for i in range(pred_step):
            hidden, cell = self.lstm(hidden, [hidden, cell])
            reward = self.reward_layer(torch.cat([hidden, steps*(i+1)], dim=1))
            rewards.append(reward.view(-1,1).unsqueeze(1))
            pred = torch.nn.Tanh()(self.fc2(hidden)) * 3
            this_value = self.value_fc(pred)
            values.append(this_value.view(-1,1).unsqueeze(1))
            preds.append(pred.unsqueeze(1))
            if i < pred_step-1:
                hidden_state = torch.cat([hidden, self.act_fc(act_sequence[:, i, :]), latent], dim=1)
                hidden = F.relu(self.fc(hidden_state))
            else:
                this_final_rewards = self.reward_layer(torch.cat([hidden, steps*(i+1)], dim=1))
        preds = torch.cat(preds, dim=1)
        rewards = torch.cat(rewards, dim=1)
        values = torch.cat(values, dim=1)
        if self.args.method2:
            rewards[:,-1, :] = rewards[:,-1, :]+ 0.9 * this_final_rewards
            final_rewards = Variable(torch.zeros_like(rewards)).to('cuda')
            final_rewards = final_rewards + rewards
            for i in range(pred_step-1):
                for j in range(i+1, pred_step):
                    final_rewards[:,i, :] += rewards[:, j, :] * (0.9 ** (j-i))
            return preds, final_rewards, values
        return preds, rewards, values
