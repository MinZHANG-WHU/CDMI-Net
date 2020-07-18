# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

from __future__ import print_function
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from mil_model import Attention
from mil_dataloader import CDBags
import time
import accuracy as acc
import argparse


def test(model, test_loader, args):
    model.eval()
    gts = []
    pred = []
    step = 0
    all_size = len(test_loader)
    time_start_all = time.time()
    time_start = time.time()
    for batch_idx, (data1, data2, label, file_name) in enumerate(test_loader):
        step = step + 1
        gts.append(label[0].numpy()[0])
        data_v_1 = Variable(data1)
        data_v_2 = Variable(data2)
        data_v_1 = data_v_1.cuda()
        data_v_2 = data_v_2.cuda()

        pred_prob, pred_label, attention_weights = model.eval_img(
            data_v_1, data_v_2)
        pred.append(pred_label[0])
        if step % args.disp == 0:
            time_end = time.time()
            print('Test step:{}/{}, Time {:.2f}'.format(
                step, all_size, time_end - time_start))
            time_start = time.time()

    time_end_all = time.time()
    print('All time {:.2f}'.format(time_end_all - time_start_all))
    hist = acc.hist(gts, pred)
    acc.evaluation_print(hist)


def train(model, args):
    args_gpu = not args.no_gpu and torch.cuda.is_available()

    if args_gpu:
        torch.cuda.manual_seed(args.seed)
        print('Using GPU')
    else:
        torch.manual_seed(args.seed)
        print('Using CPU')

    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args_gpu else {}

    print('Load training dataset')
    train_loader = data_utils.DataLoader(CDBags(data_dir=args.data_dir,
                                                seed=args.seed,
                                                train=True),
                                         batch_size=1,
                                         shuffle=True,
                                         **loader_kwargs)

    test_loader = data_utils.DataLoader(CDBags(data_dir=args.data_dir,
                                               seed=args.seed,
                                               train=False),
                                        batch_size=1,
                                        shuffle=False,
                                        **loader_kwargs)

    print('Init model')
    if args_gpu:
        model.cuda()
    # model.print_size()

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(
            0.9, 0.999), weight_decay=args.decay)

    train_loss = 0.
    train_error = 0.
    all_size = len(train_loader)

    step = 0
    train_loss_t = 0
    train_error_t = 0

    time_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data1, data2, label,
                        file_name) in enumerate(train_loader):
            bag_label = label[0]

            data_v_1 = Variable(data1)
            data_v_2 = Variable(data2)
            if args_gpu:
                data_v_1 = data_v_1.cuda()
                data_v_2 = data_v_2.cuda()
                bag_label = bag_label.cuda()

            # reset gradients
            optimizer.zero_grad()
            # calculate loss and metrics
            loss, attention_weights, error = model.calculate_loss(
                data_v_1, data_v_2, bag_label)

            it_loss = loss.data[0].cpu().numpy()[0, 0]
            it_error = error[0]

            # epoch loss
            train_loss += it_loss
            train_error += it_error

            # disp loss
            train_loss_t += it_loss
            train_error_t += it_error

            step = step + 1

            # backward pass
            loss.backward()
            # step
            optimizer.step()

            if step % args.disp == 0:
                train_loss_t = train_loss_t / args.disp
                train_error_t = train_error_t / args.disp
                time_end = time.time()
                print('Epoch:{},{}/{}, Loss: {:.4f}, Train error: {:.4f}, Time {:.2f}'.format(
                    epoch, step, all_size, train_loss_t, train_error_t, time_end - time_start))
                time_start = time.time()
                train_loss_t = 0
                train_error_t = 0

        # calculate loss and error for epoch
        train_loss = train_loss / len(train_loader)
        train_error = train_error / len(train_loader)

        path = '{}/cdminet_epoch_{}.pt'.format(args.weight_dir, epoch)
        torch.save(model.state_dict(), path)
        msg = 'Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(
            epoch, train_loss, train_error)

        test(model, test_loader, args)
        print(msg)


if __name__ == "__main__":
    '''
    python mil_train.py --data_dir DATA_DIR --weight_dir WEIGHT_DIR

    '''
    args = argparse.ArgumentParser(description='Start training stage ...')
    args.add_argument('--data_dir', required=True, help='Training set dir.')
    args.add_argument('--weight_dir', required=True, help='Check point dir.')
    args.add_argument(
        '--disp',
        type=int,
        default=100,
        help='Number of iterations for display.')
    args.add_argument('--epochs', type=int, default=30, help='Max epochs.')
    args.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    args.add_argument(
        '--decay',
        type=float,
        default=10e-5,
        help='Weight decay.')
    args.add_argument('--seed', type=int, default=1, help='Random seed.')
    args.add_argument('--no-gpu', action='store_true', help='Using CPU.')

    model = Attention()
    train(model, args.parse_args())

    print('Done!')
