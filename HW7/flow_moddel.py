# reference from https://github.com/karpathy/pytorch-normalizing-flows
from pytorch_normalizing_flows.nflib.flows import NormalizingFlowModel, Invertible1x1Conv, ActNorm
from pytorch_normalizing_flows.nflib.spline_flows import NSF_AR, NSF_CL

import torch
from torch.utils.data import DataLoader
from datahelper import CLEVRDataset
from task_1.dataset import ICLEVRLoader
import os
import itertools
from torch.distributions import Uniform, TransformedDistribution, SigmoidTransform
from flow_moddel_train import train

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

    # Neural splines, coupling
    prior = TransformedDistribution(Uniform(torch.zeros(2), torch.ones(2)), SigmoidTransform().inv) # Logistic distribution
    nfs_flow = NSF_CL if True else NSF_AR
    flows = [nfs_flow(dim=2, K=8, B=3, hidden_dim=16) for _ in range(3)]
    convs = [Invertible1x1Conv(dim=2) for _ in flows]
    norms = [ActNorm(dim=4) for _ in flows]
    flows = list(itertools.chain(*zip(norms, convs, flows)))

    # construct the model
    model = NormalizingFlowModel(prior, flows)

    # train
    train(loader_train, model, z_dim, epochs, lr)