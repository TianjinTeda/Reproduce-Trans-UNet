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
    image = np.load('../data/Synapse/train_npz/case0006_slice103.npz')['image']
    print(image.shape)
    for i in image:
        print(i)

    image = Image.fromarray(image * 255)
    image.show()

    label = np.load('../data/Synapse/train_npz/case0006_slice103.npz')['label']
    print(label.shape)
    print(np.max(label))


    label = Image.fromarray(label*20)
    label.show()


if __name__ == "__main__":
    direct_inspect()