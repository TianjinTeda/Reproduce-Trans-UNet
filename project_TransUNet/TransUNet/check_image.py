import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

def inspect_image():
    #pred = sitk.ReadImage('../predictions/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/case0002_pred.nii.gz')
    #plt.imshow(sitk.GetArrayViewFromImage(pred)[10,:,:])
    #plt.show()
    #sitk.Show(pred, 'sample image', debugOn=True)

    #nda = sitk.GetArrayFromImage(pred)
    #imshow(nda)

    img = nib.load('../predictions/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/case0002_pred.nii.gz')
    print(img)
    print(img.header['db_name'])  # 输出nii的头文件
    width, height, queue = img.dataobj.shape
    #OrthoSlicer3D(img.dataobj).show()
    num = 1
    for i in range(0, queue, 10):
        img_arr = img.dataobj[:, :, i]
        plt.subplot(5, 4, num)
        plt.imshow(img_arr, cmap='gray')
        num += 1

    plt.show()
    img = nib.load('../predictions/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/case0002_img.nii.gz')
    print(img)
    print(img.header['db_name'])  # 输出nii的头文件
    width, height, queue = img.dataobj.shape
    # OrthoSlicer3D(img.dataobj).show()
    num = 1
    for i in range(0, queue, 10):
        img_arr = img.dataobj[:, :, i]
        plt.subplot(5, 4, num)
        plt.imshow(img_arr, cmap='gray')
        num += 1

    plt.show()
    img = nib.load('../predictions/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/case0002_gt.nii.gz')
    print(img)
    print(img.header['db_name'])  # 输出nii的头文件
    width, height, queue = img.dataobj.shape
    # OrthoSlicer3D(img.dataobj).show()
    num = 1
    for i in range(0, queue, 10):
        img_arr = img.dataobj[:, :, i]
        plt.subplot(5, 4, num)
        plt.imshow(img_arr, cmap='gray')
        num += 1

    plt.show()

if __name__ == '__main__':
    inspect_image()