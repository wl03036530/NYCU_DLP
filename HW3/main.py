import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import dataloader

def gen_dataset(train_x, train_y, test_x, test_y):
    datasets = []
    for x, y in [(train_x, train_y), (test_x, test_y)]:
        x = torch.stack(
            # convert np.ndarray to tensor
            [torch.Tensor(x[i]) for i in range(x.shape[0])]
        )
        y = torch.stack(
            # convert np.ndarray to tensor
            [torch.Tensor(y[i:i+1]) for i in range(y.shape[0])]
        )
        datasets += [TensorDataset(x, y)]
        
    return datasets

def showAccuracy(title='', **kwargs):
    fig = plt.figure(figsize=(8,4.5))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    
    for label, data in kwargs.items():
        plt.plot(
            range(1, len(data)+1), data, 
            '--' if 'test' in label else '-', 
            label=label
        )
    
    plt.legend(loc='best', fancybox=True, shadow=True)
    #plt.show()
    
    return fig

class EEGNet(nn.Module):
    def __init__(self, activation=None, dropout=0.25):
        super(EEGNet, self).__init__()
        
        # set activation function
        if not activation:
            activation = nn.ELU(alpha=1.0)
        
        # Layer 1 : firstconv
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        
        # Layer 2 : depthwiseConv
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=dropout)
        )
        
        # Layer 3 : separableConv
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout)
        )

        # Layer 4 : classify
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
        

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        # flatten
        x = x.view(-1, 736)
        x = self.classify(x)
        
        return x

from functools import reduce
class DeepConvNet(nn.Module):
    def __init__(self, activation=None, deepconv=[25,50,100,200], dropout=0.5):
        super(DeepConvNet, self).__init__()
        
        if not activation:
            activation = nn.ELU
        
        self.deepconv = deepconv

        # Layer 0
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, deepconv[0], kernel_size=(1, 5)),
            nn.Conv2d(deepconv[0], deepconv[0], kernel_size=(2,1)),
            nn.BatchNorm2d(deepconv[0], eps=1e-05, momentum=0.1),
            activation(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=dropout)
        )
        
        for idx in range(1, len(deepconv)):
            setattr(self, 'conv'+str(idx), nn.Sequential(
                nn.Conv2d(deepconv[idx-1], deepconv[idx], kernel_size=(1,5), stride=(1,1), padding=(0,0), bias=True),
                nn.BatchNorm2d(deepconv[idx], eps=1e-05, momentum=0.1),
                activation(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=dropout)
            ))
        
        
        flatten_size =  deepconv[-1] * reduce(lambda x,_: round((x-4)/2), deepconv, 750)
        self.classify = nn.Sequential(
            nn.Linear(flatten_size, 2, bias=True),
        )
    
    def forward(self, x):
        for i in range(len(self.deepconv)):
            x = getattr(self, 'conv'+str(i))(x)
        # flatten
        x = x.view(-1, self.classify[0].in_features)
        x = self.classify(x)
        return x

def runModels(
    models, train_dataset, test_dataset, epoch_size, batch_size, learning_rate, 
    optimizer = optim.Adam , criterion = nn.CrossEntropyLoss(),
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, len(test_dataset))
    
    Accs = {
    **{key+"_train" : [] for key in models},
    **{key+"_test" : [] for key in models}
    }
    
    optimizers = {
        key: optimizer(value.parameters(), lr=learning_rate) 
        for key, value in models.items()
    }
    for epoch in range(epoch_size):
        train_correct = {key:0.0 for key in models}
        test_correct = {key:0.0 for key in models}
        # training multiple model
        for idx, data in enumerate(train_loader):
            x, y = data
            inputs = x.to(device)
            labels = y.to(device).long().view(-1)
        
            for optimizer in optimizers.values():
                optimizer.zero_grad()
        
            for key, model in models.items():
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
            
                train_correct[key] += (
                    torch.max(outputs, 1)[1] == labels
                ).sum().item()
        
            for optimizer in optimizers.values():
                optimizer.step()
        
        # testing multiple model
        with torch.no_grad():
            for _, data in enumerate(test_loader):
                x, y = data
                inputs = x.to(device)
                labels = y.to(device)
        
                for key, model in models.items():
                    outputs = model.forward(inputs)
        
                    test_correct[key] += (
                        torch.max(outputs, 1)[1] == labels.long().view(-1)
                    ).sum().item()

        for key, value in train_correct.items():
            Accs[key+"_train"] += [(value*100.0) / len(train_dataset)]
    
        for key, value in test_correct.items():
            Accs[key+"_test"] += [(value*100.0) / len(test_dataset)]
        
    # epoch end
    torch.cuda.empty_cache()
    return Accs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('pytorch device : ', device)

    # data
    train_dataset, test_dataset = gen_dataset(*dataloader.read_bci_data())

    ###--- EEGNet
    print('Training & Testing EEGNet')
    models = {
        "elu" : EEGNet(nn.ELU).to(device),
        "relu" : EEGNet(nn.ReLU).to(device),
        "leaky_relu" : EEGNet(nn.LeakyReLU).to(device),
    }
    Accs = runModels(models, train_dataset, test_dataset, epoch_size=300, batch_size=64, learning_rate=1e-3)
    showAccuracy("EEGNet", **Accs).savefig('EEGNet.png')

    EEG_ReLU_ACC = max(Accs['relu_test'])
    EEG_Leakly_ReLU_ACC = max(Accs['leaky_relu_test'])
    EEG_ELU_ACC = max(Accs['elu_test'])
    # print(EEG_ReLU_ACC, EEG_Leakly_ReLU_ACC, EEG_ELU_ACC)

    ###--- DeepConvNet
    print('Training & Testing DeepConvNet')
    models = {
        "elu" : DeepConvNet(nn.ELU).to(device),
        "relu" : DeepConvNet(nn.ReLU).to(device),
        "leaky_relu" : DeepConvNet(nn.LeakyReLU).to(device),
    }
    Accs = runModels(models, train_dataset, test_dataset, epoch_size=300, batch_size=64, learning_rate=1e-3)
    showAccuracy("DeepConvNet", **Accs).savefig('DeepConvNet.png')

    DeepConvNet_ReLU_ACC = max(Accs['relu_test'])
    DeepConvNet_Leakly_ReLU_ACC = max(Accs['leaky_relu_test'])
    DeepConvNet_ELU_ACC = max(Accs['elu_test'])
    # print(DeepConvNet_ReLU_ACC, DeepConvNet_Leakly_ReLU_ACC, DeepConvNet_ELU_ACC)

    # draw table
    fig, ax =plt.subplots()
    data=[[EEG_ReLU_ACC, EEG_Leakly_ReLU_ACC, EEG_ELU_ACC],
        [DeepConvNet_ReLU_ACC, DeepConvNet_Leakly_ReLU_ACC, DeepConvNet_ELU_ACC]]
    column_labels=["ReLU", "Leaky ReLU", "ELU"]
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText = data, colLabels = column_labels, rowLabels=["EEGNet", "DeepConvNet"], loc = "center")

    plt.show()
    fig.savefig('compare.png')
