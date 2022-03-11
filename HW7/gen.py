import torch
import torch.nn as nn
import numpy as np
import os
from task_1.evaluator import evaluation_model
from task_1.dataset import ICLEVRLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gen_pic(generator, z_dim):
    # result path
    task1_result_path = os.path.join(os.getcwd(), 'task1_result')
    if not os.path.exists(task1_result_path):
        os.makedirs(task1_result_path)
    
    generator_path = os.path.join(os.getcwd(), 'task1_model')
    generator.load_state_dict(torch.load(os.path.join(generator_path, "epoch137score_0.6527777777777778.pkl")))

    # testing data (condition)
    data_detail = ICLEVRLoader(os.path.join(os.getcwd(), 'task_1'), mode='test')
    test_conditions = torch.FloatTensor(data_detail.label_list).to(device)
    fixed_z = torch.randn(len(test_conditions), z_dim).to(device)

    fake_img = generator(fixed_z, test_conditions)
    eval_score = evaluation_model()
    score = eval_score.eval(fake_img, test_conditions)
    print('score: '+str(score))

    plt.imshow(np.transpose(vutils.make_grid(fake_img.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(os.path.join(task1_result_path, 'new_result'))