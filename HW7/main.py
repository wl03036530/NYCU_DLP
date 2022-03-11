import os
import torch
from torch.utils.data import DataLoader
from datahelper import CLEVRDataset
from task_1.dataset import ICLEVRLoader
from model import Generator,Discriminator
from train import train
from gen import gen_pic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 100
c_dim = 64*64
image_shape = (64,64,3)
epochs = 200
lr = 0.0002
batch_size = 64


if __name__=='__main__':

    # load training data
    data_detail = ICLEVRLoader(os.path.join(os.getcwd(), 'task_1'))
    image_path = os.path.join(os.getcwd(), 'task_1', 'images')
    dataset_train = CLEVRDataset(image_path, data_detail.img_list, data_detail.label_list)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    # # create generate & discriminator
    generator = Generator(z_dim,c_dim).to(device)
    discrimiator = Discriminator(image_shape).to(device)

    # train
    # train(loader_train, generator, discrimiator, z_dim, epochs, lr)

    # gen
    gen_pic(generator, z_dim)