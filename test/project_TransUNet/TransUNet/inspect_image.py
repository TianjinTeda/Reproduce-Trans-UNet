from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from PIL import Image
import torch
import numpy as np


unloader = transforms.ToPILImage()
def worker_init_fn(worker_id):
    random.seed(42 + worker_id)


def inspect():
    '''
    Inspect the image and mask from the torch Tensor
    '''
    db_train = Synapse_dataset(base_dir='../data/Synapse/train_npz', list_dir='lists/lists_Synapse', split="train", transform=transforms.Compose([RandomGenerator(output_size=[224, 224])]))
    trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
    i = 1
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        if i == 1:
            print(torch.max(image_batch))
            print(torch.min(image_batch))
            image_batch = unloader(image_batch[0])
            image_batch.show()
            label_batch = unloader(label_batch[0] * 1.0)
            label_batch.show()
        i += 1


def direct_inspect():
    '''
        Inspect the image and mask directly from the numpy files
    '''
    '''
    image = np.load('../data/nucleus/train_npz/nucleus_3.npz')['image']
    image = np.transpose(image, (2, 1, 0))
    print(image.shape)
    print(np.min(image))
    print(np.max(image))
    image = Image.fromarray(np.uint8(image * 255), 'RGB')
    image.show()

    image = np.load('../data/nucleus/train_npz/nucleus_3.npz')['label']
    #image = np.transpose(image, (2, 1, 0))
    print(image.shape)
    print(np.min(image))
    print(np.max(image))
    image = Image.fromarray(np.uint8(image * 255))
    image.show()

    '''
    image = np.load('../predictions/result_image_brain/brain_46630.npz_gt_.npy')
    #print(image.shape)
    #print(np.min(image))
    #print(np.max(image))
    new_image = np.zeros((224, 224, 3))
    for i in range(0, 224):
        for j in range(0, 224):
            if image[i][j] == 0:
                new_image[i][j][0] = 0
                new_image[i][j][1] = 0
                new_image[i][j][2] = 0
            elif image[i][j] == 1:
                new_image[i][j][0] = 255
                new_image[i][j][1] = 0
                new_image[i][j][2] = 0
            elif image[i][j] == 2:
                new_image[i][j][0] = 0
                new_image[i][j][1] = 255
                new_image[i][j][2] = 0
            elif image[i][j] == 3:
                new_image[i][j][0] = 0
                new_image[i][j][1] = 0
                new_image[i][j][2] = 255

    image = Image.fromarray(np.uint8(new_image))
    image.show()

    label = np.load('../predictions/result_image_brain/brain_46630.npz_pred_.npy')
    new_image = np.zeros((224, 224, 3))
    for i in range(0, 224):
        for j in range(0, 224):
            if label[i][j] == 0:
                new_image[i][j][0] = 0
                new_image[i][j][1] = 0
                new_image[i][j][2] = 0
            elif label[i][j] == 1:
                new_image[i][j][0] = 255
                new_image[i][j][1] = 0
                new_image[i][j][2] = 0
            elif label[i][j] == 2:
                new_image[i][j][0] = 0
                new_image[i][j][1] = 255
                new_image[i][j][2] = 0
            elif label[i][j] == 3:
                new_image[i][j][0] = 0
                new_image[i][j][1] = 0
                new_image[i][j][2] = 255

    label = Image.fromarray(np.uint8(new_image))
    label.show()



if __name__ == "__main__":
    direct_inspect()