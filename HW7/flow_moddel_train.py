import torch
import torch.nn as nn
import numpy as np
from task_1.dataset import ICLEVRLoader
import os
from task_1.evaluator import evaluation_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataloader, model, z_dim, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(1, 1+epochs):
        # train
        for i, (img, condition) in enumerate(dataloader):
            model.train()

            img = img.to(device)
            condition = condition.float().to(device)

            zs, prior_logprob, log_det = model(torch.cat((img, condition), dim=1))
            logprob = prior_logprob + log_det
            loss = -torch.sum(logprob) # NLL

            model.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'epoch{epoch} {i}/{len(dataloader)}  loss: {loss.item():.3f}')
        
        # evaluate
        model.eval()
        eval_score = evaluation_model()

        # with torch.no_grad():
        #     fake_img = generator(fixed_z, test_conditions)
        # score = eval_score.eval(fake_img, test_conditions)
