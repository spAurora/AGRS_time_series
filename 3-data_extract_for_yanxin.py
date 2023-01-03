# -*- coding: utf-8 -*-
"""
原始tif影像转化为csv文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Created on Thu Jul 29 11:25:34 2021
Reconstructed by wHy 2023.01.01
"""

import os,sys
from osgeo import gdal
from osgeo import osr
import numpy as np
from osgeo import ogr
import linecache
from skimage import io
import csv
import math

def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs

def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]

def imagexy2geo(dataset, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py

def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
        return
    im_width = dataset.RasterXSize # 栅格矩阵的列数
    im_height = dataset.RasterYSize # 栅格矩阵的行数
    im_bands = dataset.RasterCount # 波段数
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) # 获取数据
    


if __name__ == '__main__':

    img_path = r'E:\yanxin\data/岩心高光谱图像_Export.tif' # 影像路径
    save_path = r'E:\yanxin\data/sr_img.csv' # 输出路径
    
    img = io.imread(img_path)
    #img = img.transpose(1, 2, 0)
    height = img.shape[0]
    width = img.shape[1]
    channel = img.shape[2]

    print(height, width, channel)

    middle = np.zeros([height*width, channel])
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, channel):
                middle[i*width + j][k] = img[i][j][k]

    with open(save_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(middle)

    
    
    
    
    
    
    
    
    