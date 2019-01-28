import os
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

from models import LSTM
from dataset import NaiveDataset, AdvancedDataset


def load_model(model, path, strict=False):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key[:6] == 'module':
            name = key[7:]  # remove 'module'
        else:
            name = key
        new_state_dict[name] = val
    model.load_state_dict(new_state_dict, strict=strict)
    all_keys = set(new_state_dict.keys())
    actual_keys = set(model.state_dict().keys())
    missing_keys = actual_keys - all_keys
    for k in missing_keys:
        print(k)


def train(train_dataloader, dev_dataloader, model, optimizer, lr_scheduler, args):
    model.train()
    if args.loss_function == 'L1Loss':
        criterion = nn.L1Loss()
    elif args.loss_function == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError
    batch_size = args.batch_size
    print('batch_size: {}'.format(batch_size))
    log_f_name = '%s/log.txt' % (args.output_dir)
    log_f = open(log_f_name, "w")

    list_loss = []
    list_accuracy = []
    for epoch in range(args.epoch):
        #lr_scheduler.step()
        count = 0
        total_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            feature, label = data
            feature, label = feature.cuda(), label.cuda()
            cur_length = label.shape[0]
            output = model(feature)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            total_loss += loss.item()
            print('[epoch%d: batch%d], train loss: %f' % (epoch, idx, loss.item()))
        list_loss.append(total_loss/count)
        print('[epoch%d], avg train loss: %f' % (epoch, total_loss/count), file=log_f)
        print('[epoch%d], avg train loss: %f' % (epoch, total_loss/count))

        if epoch % 2 == 0:
            torch.save(model.state_dict(), "%s/epoch_%d.pth" % (args.output_dir, epoch))
    list_loss_np = np.array(list_loss)
    np.save('%s/loss' % (args.output_dir), list_loss_np)
    log_f.close()


def validate(test_dataloader, model):
    total_correct = 0
    total = 0
    for idx, data in enumerate(test_dataloader):
        feature, label = data
        feature, label = feature.cuda(), label.cuda()
        cur_length = label.shape[0]
        output = model(feature)

        output_choice = output.data.max(dim=1)[1]
        correct = output_choice.eq(label).sum().cpu().numpy()
        total += cur_length
        total_correct += correct
    print('test accuracy: %f' % (correct * 1.0 / total))

def main(args):
    if args.model == 'LSTM':
        model = LSTM(input_dim=args.input_dim, lstm_hidden_dim=args.lstm_hidden_dim, time_step=args.time_step)
    else:
        raise ValueError
    model.cuda()

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'SGD_momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.999, eps=1e-8, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    else:
        raise ValueError

    lr_scheduler = None
    if args.load_path:
        if args.recover:
            load_model(model, args.load_path, strict=True)
            print('load model state dict in {}'.format(args.load_path))

    map_file_path = 'divide.csv'
    data_file_path = 'processed_data.txt'
    train_set = NaiveDataset(data_file_path, map_file_path)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)

    if args.evaluate:
        validate(test_dataloader, model)
        return

    train(train_dataloader, train_dataloader, model, optimizer, lr_scheduler, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mcm LSTM')
    parser.add_argument('--load_path', default='./experiment/naive/epoch_14.pth', type=str)
    parser.add_argument('--recover', default=True, type=bool)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--optim', default='SGD_momentum', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--input_dim', default=8, type=int)
    parser.add_argument('--lstm_hidden_dim', default=[30, 45, 40, 30], type=list)
    parser.add_argument('--time_step', default=3, type=int)
    parser.add_argument('--model', default='LSTM', type=str)
    parser.add_argument('--dataset', default='NaiveDataset', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--loss_function', default='MSELoss', type=str)
    parser.add_argument('--output_dir', default='./experiment/naive_l2', type=str)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    print(args)
    main(args)
