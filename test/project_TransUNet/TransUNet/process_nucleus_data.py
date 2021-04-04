from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize

def process_data():
    data_path = '../stage1_train/'
    image_ids = next(os.walk(data_path))[1]

    for n, id_ in tqdm(enumerate(image_ids), total=len(image_ids)):
        #sample = {}
        path = data_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:3]
        img = resize(img, (224, 224), mode='constant', preserve_range=True)
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0
        print(img.shape)
        mask = np.zeros((224, 224), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = resize(mask_, (224, 224), mode='constant', preserve_range=True)
            mask = np.maximum(mask, mask_)
        mask = mask / 255.0
        #sample['image'] = img
        #sample['label'] = mask
        #sample = {
        #    'image': img,
        #    'label': mask
        #}
        np.savez('../data/nucleus/' + 'nucleus_' + str(n), image=img, label=mask)

def create_list():
    with open('lists/lists_nucleus/train.txt', 'w') as f:
        for i in range(0, 600):
            f.write('nucleus_' + str(i) + '\n')


if __name__ == '__main__':
    process_data()