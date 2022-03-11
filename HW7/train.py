import torch
import torch.nn as nn
import numpy as np
from task_1.dataset import ICLEVRLoader
import os
from task_1.evaluator import evaluation_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataloader, generator, discrimiator, z_dim, epochs, lr):

    Criterion=nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr)
    optimizer_d = torch.optim.Adam(discrimiator.parameters(), lr)

    # testing data (condition)
    data_detail = ICLEVRLoader(os.path.join(os.getcwd(), 'task_1'), mode='test')
    test_conditions = torch.FloatTensor(data_detail.label_list).to(device)
    fixed_z = torch.randn(len(test_conditions), z_dim).to(device)
    
    # modle path
    task1_model_path = os.path.join(os.getcwd(), 'task1_model')
    if not os.path.exists(task1_model_path):
        os.makedirs(task1_model_path)

    for epoch in range(1, 1+epochs):
        total_loss_g = 0
        total_loss_d = 0
        for i, (img, condition) in enumerate(dataloader):
            generator.train()
            discrimiator.train()
            batch_size = len(img)
            img = img.to(device)
            condition = condition.float().to(device)

            # true / false map
            real = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)

            optimizer_d.zero_grad()
            
            # real image
            pred = discrimiator(img, condition)
            loss_real = Criterion(pred, real)

            # fake image
            z = torch.randn(batch_size, z_dim).to(device)
            fake_img = generator(z, condition)
            pred = discrimiator(fake_img, condition)
            loss_fake = Criterion(pred, fake)

            # bp
            loss_d = loss_fake + loss_real
            loss_d.backward()
            optimizer_d.step()

            avg_loss_g = 0
            for _ in range(5):
                optimizer_g.zero_grad()
                z = torch.randn(batch_size, z_dim).to(device)
                fake_img = generator(z, condition)
                pred = discrimiator(fake_img, condition)
                loss_g = Criterion(pred, real)
                avg_loss_g += loss_g

                # bp
                loss_g.backward()
                optimizer_g.step()
            
            avg_loss_g /= 5

            print(f'epoch{epoch} {i}/{len(dataloader)}  loss_g: {loss_g.item():.3f}  loss_d: {loss_d.item():.3f}')
            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()

        # evaluate
        generator.eval()
        discrimiator.eval()
        eval_score = evaluation_model()
        with torch.no_grad():
            fake_img = generator(fixed_z, test_conditions)
        score = eval_score.eval(fake_img, test_conditions)

        if score > 0.7:
            torch.save(generator.state_dict(), os.path.join(task1_model_path, 'epoch'+str(epoch)+'score_'+str(score)+'.pkl'))
        
        print(f'avg loss_g: {total_loss_g/len(dataloader):.3f}  avg_loss_d: {total_loss_d/len(dataloader):.3f}')
        print(f'testing score: {score:.2f}')
        print('---------------------------------------------')
