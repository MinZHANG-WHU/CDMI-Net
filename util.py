# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

import gdal
import numpy as np
import cv2
from PIL import Image


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float32)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float32)
    t_quantiles /= t_quantiles[-1]
    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[bin_idx].reshape(oldshape).astype(np.float32)


def read_image(path):
    im = Image.open(path)
    im = np.asarray(im)
    return np.asarray(im, dtype=np.float32)


def save_map(path, data):
    data = np.asarray(data, dtype=np.uint8)
    cv2.imwrite(path, data)


def read_tiff(path):
    dataset = gdal.Open(path)
    if dataset is None:
        print(path + "can't open")
        return
    return dataset


def create_tiff(path, info_path, datatype=gdal.GDT_Byte):
    dataset = gdal.Open(info_path)
    if dataset is None:
        print(info_path + "can't open")
        return

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = 1

    driver = gdal.GetDriverByName("GTiff")
    out_dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if out_dataset:
        out_dataset.SetGeoTransform(im_geotrans)
        out_dataset.SetProjection(im_proj)
    return out_dataset


def read_block(dataset, x, y, dim):
    im_width = dataset.RasterXSize  #
    im_height = dataset.RasterYSize  #
    dimx = dim
    dimy = dim
    if x + dim > im_width:
        dimx = im_width - x
    if y + dim > im_height:
        dimy = im_height - y
    im_data = dataset.ReadAsArray(x, y, dimx, dimy)  #
    im_data = np.asarray(im_data, dtype=np.float32)  # b,h,w
    if len(im_data.shape) == 3:
        im_data = im_data[0:3, :, :]
        im_data = im_data.transpose((1, 2, 0))  # h,w,b
        im_data_out = np.zeros((dim, dim, 3), dtype=np.float32)
        im_data_out[0:dimy, 0:dimx] = im_data
        return im_data_out
    else:
        im_data_out = np.zeros((dim, dim), dtype=np.float32)
        im_data_out[0:dimy, 0:dimx] = im_data
        return im_data_out


def write_block(dataset, data, x, y, dim):
    data = np.asarray(data, dtype=np.uint8)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    dim = data.shape[0]
    if x + dim > im_width:
        data = data[:, 0:im_width - x]
    if y + dim > im_height:
        data = data[im_height - y:0, :]

    dataset.GetRasterBand(1).WriteArray(data, x, y)
    dataset.FlushCache()


def FLSE(I0, mask, sigma=3, gaussian_size=5, delt=25, iter=300):
    '''
    :param I0:
    :param mask:
    :param sigma:
    :param gaussian_size:
    :param delt: time step
    :param iter: teration number
    :return:
    '''
    if len(I0.shape) == 3:
        I = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    else:
        I = I0
    I = I.astype(np.double)

    I = 1.0 / 255 * I
    phi = I.copy()
    L = mask < 0.5  # out
    R = mask > 0.5  # in
    phi[L] = -1.0
    phi[R] = 1.0
    for n in range(1, iter + 1):
        phi_L0 = phi.copy()
        phi_GE0 = phi.copy()
        phi_L0[phi < 0] = 1
        phi_L0[phi >= 0] = 0
        phi_GE0[phi < 0] = 0
        phi_GE0[phi >= 0] = 1

        dx = cv2.Sobel(phi, cv2.CV_64F, 1, 0, ksize=1)
        dy = cv2.Sobel(phi, cv2.CV_64F, 0, 1, ksize=1)
        dx = cv2.convertScaleAbs(dx)
        dy = cv2.convertScaleAbs(dy)

        sum_phi_L0 = np.sum(phi_L0)
        if sum_phi_L0 == 0:
            c1 = 0
        else:
            c1 = np.sum(I * phi_L0) / sum_phi_L0
        sum_phi_GE0 = np.sum(phi_GE0)
        if sum_phi_GE0 == 0:
            c2 = 0
        else:
            c2 = np.sum(I * phi_GE0) / sum_phi_GE0
        F = (c2 - c1) * (2 * I - c1 - c2)
        F = F / (np.max(np.abs(F)))
        phi = phi + delt * F * np.sqrt(dx ** 2 + dy ** 2)

        phi_L0[phi < 0] = 1
        phi_L0[phi >= 0] = 0
        phi_GE0[phi < 0] = 0
        phi_GE0[phi >= 0] = 1

        phi = phi_GE0 - phi_L0
        phi = cv2.GaussianBlur(
            phi, (gaussian_size, gaussian_size), sigma, sigma)

    # post processing
    phi_L0[phi < 0] = 1
    phi_L0[phi >= 0] = 0
    phi_GE0[phi < 0] = 0
    phi_GE0[phi >= 0] = 1

    phi = phi_GE0 - phi_L0
    phi[phi < 0] = 0
    return phi
