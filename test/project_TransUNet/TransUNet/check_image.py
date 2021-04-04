import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
from PIL import Image
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
    print(img.header['db_name'])
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
    print(img.header['db_name'])
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
    print(img.header['db_name'])
    width, height, queue = img.dataobj.shape
    # OrthoSlicer3D(img.dataobj).show()
    num = 1
    for i in range(0, queue, 10):
        img_arr = img.dataobj[:, :, i]
        plt.subplot(5, 4, num)
        plt.imshow(img_arr, cmap='gray')
        num += 1

    plt.show()


def inspect_prediction():
    pred = np.load('../predictions/result_image/case0035_gt_70.npy')
    pred = np.repeat(pred[:, :, np.newaxis], 3, axis=2)
    pred = np.uint8(pred)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            for k in range(pred.shape[2]):
                if pred[i][j][k] == 1:
                    pred[i][j][0] = 255
                    pred[i][j][1] = 0
                    pred[i][j][2] = 0
                elif pred[i][j][k] == 2:
                    pred[i][j][0] = 0
                    pred[i][j][1] = 255
                    pred[i][j][2] = 0
                elif pred[i][j][k] == 3:
                    pred[i][j][0] = 0
                    pred[i][j][1] = 0
                    pred[i][j][2] = 255
                elif pred[i][j][k] == 4:
                    pred[i][j][0] = 255
                    pred[i][j][1] = 255
                    pred[i][j][2] = 0
                elif pred[i][j][k] == 5:
                    pred[i][j][0] = 0
                    pred[i][j][1] = 255
                    pred[i][j][2] = 255
                elif pred[i][j][k] == 6:
                    pred[i][j][0] = 255
                    pred[i][j][1] = 0
                    pred[i][j][2] = 255
                elif pred[i][j][k] == 7:
                    pred[i][j][0] = 255
                    pred[i][j][1] = 153
                    pred[i][j][2] = 51
                elif pred[i][j][k] == 8:
                    pred[i][j][0] = 255
                    pred[i][j][1] = 51
                    pred[i][j][2] = 255
                elif pred[i][j][k] == 9:
                    pred[i][j][0] = 123
                    pred[i][j][1] = 54
                    pred[i][j][2] = 198
    img = Image.fromarray(pred, 'RGB')
    print(img)
    img.show()

if __name__ == '__main__':
    inspect_prediction()