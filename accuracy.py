# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

from __future__ import print_function
from __future__ import division
import numpy as np
from datetime import datetime


def hist(gt_data, pred_data):
    gt_data = np.asarray(gt_data)
    pred_data = np.asarray(pred_data)
    gt_data[gt_data > 0.5] = 1
    gt_data[gt_data < 1] = 0
    pred_data[pred_data > 0.5] = 1
    pred_data[pred_data < 1] = 0
    hist = np.zeros((2, 2))
    tp = np.count_nonzero((gt_data == pred_data) & (gt_data > 0))
    tn = np.count_nonzero((gt_data == pred_data) & (gt_data == 0))
    fp = np.count_nonzero(gt_data < pred_data)
    fn = np.count_nonzero(gt_data > pred_data)
    hist[0, 0] = tp
    hist[1, 1] = tn
    hist[0, 1] = fp
    hist[1, 0] = fn
    return hist


def evaluation_print(hist):
    """
                     GT:Changed, Unchanged
      Predicted-Changed:  TP   ,     FP    , b1
    Predicted-Unchanged:  FN   ,     TN    , b2
                          a1   ,     a2
    """
    tp = hist[0, 0]
    fp = hist[0, 1]  # gt->0,predict->1, false alarms
    fn = hist[1, 0]  # gt->1,predict->0, missed detections
    if tp == 0:
        recall = 0
        precision = 0
        f1measure = 0
    else:
        recall = tp * 1.0 / (tp + fn)
        precision = tp * 1.0 / (tp + fp)
        f1measure = 2 * recall * precision / (recall + precision)

    print('>>>', datetime.now(), "---------Accuracy-------")
    print('>>>', datetime.now(), ("      recall:", recall))
    print('>>>', datetime.now(), ("   precision:", precision))
    print('>>>', datetime.now(), ("  f1-measure:", f1measure))
    print('>>>', datetime.now(), "----------------------")
