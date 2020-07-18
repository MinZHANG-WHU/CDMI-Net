# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

import os
import torch
from torch.autograd import Variable
from mil_model import Attention
import numpy as np
import util
import accuracy as acc
from tqdm import tqdm
import time
import argparse


def eval_img(model, args):
    time_start_all = time.time()
    model.load_state_dict(torch.load(args.weight))
    # model.print_size()
    if not args.no_gpu:
        model.cuda()

    dim = 112
    t1_path = args.t1
    t2_path = args.t2
    gt_path = args.gt

    out_path_flse = os.path.join(args.save_dir, 'pixel_flse.tif')
    out_path_score = os.path.join(args.save_dir, 'pixel_score.tif')
    out_path_bm = os.path.join(args.save_dir, 'pixel_bm.tif')
    out_path_scene = os.path.join(args.save_dir, 'scene.tif')
    out_dataset_flse = util.create_tiff(out_path_flse, t1_path)
    out_dataset_score = util.create_tiff(out_path_score, t1_path)
    out_dataset_bm = util.create_tiff(out_path_bm, t1_path)
    out_dataset_scene = util.create_tiff(out_path_scene, t1_path)

    t1 = util.read_tiff(t1_path)
    t2 = util.read_tiff(t2_path)
    w = t1.RasterXSize
    h = t1.RasterYSize
    h_batch = int(h / dim)
    w_batch = int(w / dim)
    all_count = h_batch * w_batch
    hist = np.zeros((2, 2))
    if len(gt_path) > 1:
        gt = util.read_tiff(gt_path)

    for index in tqdm(range(all_count)):
        i = int(index / w_batch)  # row
        j = index % w_batch  # col
        x = j * dim
        y = i * dim
        t1_b = util.read_block(t1, x, y, dim)
        t2_b = util.read_block(t2, x, y, dim)
        t2_b = util.hist_match(t2_b, t1_b)

        if len(gt_path) > 1:
            gt_b = util.read_block(gt, x, y, dim)
            gt_b[gt_b < 255] = 0
            #gt_b[gt_b == 255] = 0
            #gt_b[gt_b > 0] = 255

        data1 = t1_b.transpose((2, 0, 1))
        data2 = t2_b.transpose((2, 0, 1))

        data1 = data1[np.newaxis, ...]
        data2 = data2[np.newaxis, ...]

        data_v_1 = Variable(torch.from_numpy(data1))
        data_v_2 = Variable(torch.from_numpy(data2))

        if not args.no_gpu:
            data_v_1 = data_v_1.cuda()
            data_v_2 = data_v_2.cuda()

        model.train()
        pred_prob, pred_label, attention_weights = model.eval_img(
            data_v_1, data_v_2)
        model.eval()
        if pred_label[0] > 0.5:
            pred_label = 'P'
            bmm = np.ones((dim, dim)) * 255
            weight = attention_weights.data[0].cpu().detach().numpy()
            weight = weight.reshape((dim, dim))
            cmm = weight * 255.0 / np.max(weight)
            bm = cmm.copy()
            bm[bm < 128] = 0
            bm[bm > 0] = 255
            cva = np.abs(t1_b - t2_b)
            cva = np.power(cva, 2)
            cva = np.sum(cva, axis=2)
            cva = cva / 3.0
            cva = np.sqrt(cva)
            flse = util.FLSE(
                cva,
                bm,
                args.sigma,
                args.gaussian,
                args.delt,
                args.iter)
            flse = np.asarray(flse, dtype=np.uint8)
            flse[flse > 0] = 255
            flse[flse <= 0] = 0

            sub_dir = os.path.join(args.save_dir)
            b1_path = os.path.join(sub_dir, "r{0}_c{1}_b1.tif".format(i, j))
            util.save_map(b1_path, t1_b)

            b2_path = os.path.join(sub_dir, "r{0}_c{1}_b2.tif".format(i, j))
            util.save_map(b2_path, t2_b)
            if len(gt_path) > 1:
                bg_path = os.path.join(
                    sub_dir, "r{0}_c{1}_gt.tif".format(i, j))
                util.save_map(bg_path, gt_b)

            bm_path = os.path.join(sub_dir, "r{0}_c{1}_bm.tif".format(i, j))
            util.save_map(bm_path, bm)

            score_path = os.path.join(
                sub_dir, "r{0}_c{1}_score.tif".format(i, j))
            util.save_map(score_path, cmm)

            flse_path = os.path.join(
                sub_dir, "r{0}_c{1}_flse.tif".format(i, j))
            util.save_map(flse_path, flse)

        else:
            pred_label = 'N'
            cmm = np.zeros((dim, dim))
            bm = np.zeros((dim, dim))
            flse = np.zeros((dim, dim))
            bmm = np.zeros((dim, dim))

        util.write_block(out_dataset_flse, flse, x, y, dim)
        util.write_block(out_dataset_score, cmm, x, y, dim)
        util.write_block(out_dataset_bm, bm, x, y, dim)
        util.write_block(out_dataset_scene, bmm, x, y, dim)
        if len(gt_path) > 1:
            gt_pixel_count = np.count_nonzero(gt_b)
            if gt_pixel_count > 30:
                if pred_label == 'P':
                    hist[0, 0] = hist[0, 0] + 1.0  # TP
                else:
                    hist[1, 0] = hist[1, 0] + 1.0  # FN
            else:
                if pred_label == 'P':
                    hist[0, 1] = hist[0, 1] + 1.0  # FP
                else:
                    hist[1, 1] = hist[1, 1] + 1.0  # TN

    del out_dataset_flse
    del out_dataset_score
    del out_dataset_bm
    del out_dataset_scene

    if len(gt_path) > 1:
        time_end_all = time.time()
        print('All time {:.2f}'.format(time_end_all - time_start_all))
        print("CDMI-Net: Scene-based accuracy")
        acc.evaluation_print(hist)

        gt_data = util.read_image(gt_path)
        gt_data[gt_data == 255] = 1
        #gt_data[gt_data == 255] = 0
        pred_data = util.read_image(out_path_flse)
        acc_matrix = acc.hist(gt_data, pred_data)
        print("CDMI-Net: Pixel-based accuracy")
        acc.evaluation_print(acc_matrix)


if __name__ == "__main__":
    '''
    python mil_infer.py --t1 T1_IMAGE_PATH --t2 T2_IMAGE_PATH --weight CHECK_POINT_PATH--save-dir OUTPUT_PATH --gt GT_PATH

    '''
    args = argparse.ArgumentParser(description='Start inference stage ...')
    args.add_argument('--t1', required=True, help='First period image path.')
    args.add_argument('--t2', required=True, help='Second period image path.')
    args.add_argument('--gt', help='Gound truth path.', default='')
    args.add_argument('--weight', required=True, help='Check point path.')
    args.add_argument('--save-dir', required=True, help='Output dir.')

    args.add_argument(
        '--sigma',
        type=int,
        default=1,
        help='Parameter [sigma] of FLSE.')
    args.add_argument('--gaussian', type=int, default=9,
                      help='Parameter [gaussian_size] of FLSE.')
    args.add_argument(
        '--delt',
        type=int,
        default=8,
        help='Parameter [sigma] of FLSE.')
    args.add_argument(
        '--iter',
        type=int,
        default=20,
        help='Parameter [iter] of FLSE.')
    args.add_argument('--no-gpu', action='store_true', help='Using CPU.')

    model = Attention()
    eval_img(model, args.parse_args())
    print('Done!')
