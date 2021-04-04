from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import h5py

def process_data():
    data_path = '../../data/brain/'
    n = 0    
    for file in os.listdir(data_path):
        f = h5py.File(data_path+file)
        img = f['image'][:]
        mask = f['mask'][:]
        img = img[:, :, :3]
        img = resize(img, (224, 224), mode='constant', preserve_range=True)
        img = np.transpose(img, (2, 0, 1))
        new_mask = np.zeros((240, 240))
        for i in range(0, 240):
            for j in range(0, 240):
                if mask[i][j][0] == 0 and mask[i][j][1] == 0 and mask[i][j][2] == 0:
                    new_mask[i][j] = 0.0
                elif mask[i][j][0] == 1 and mask[i][j][1] == 0 and mask[i][j][2] == 0:
                    new_mask[i][j] = 1.0
                elif mask[i][j][0] == 0 and mask[i][j][1] == 1 and mask[i][j][2] == 0:
                    new_mask[i][j] = 2.0
                elif mask[i][j][0] == 0 and mask[i][j][1] == 0 and mask[i][j][2] == 1:
                    new_mask[i][j] = 3.0
        new_mask = resize(new_mask, (224, 224), mode='constant', preserve_range=True)
        np.savez('../../data/brain/' + 'brain_' + str(n), image=img, label=new_mask)
        n += 1
        print(n)


def create_list():
    with open('../lists/lists_brain/test_vol.txt', 'w') as f:
        for i in range(46500, 57195):
            f.write('brain_' + str(i) + '.npz\n')


if __name__ == '__main__':
    create_list()