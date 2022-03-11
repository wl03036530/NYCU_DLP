import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import transforms
import torchvision.models
import os
import pyprind

from dataloader import RetinopathyLoader

# load data
train_dataset = RetinopathyLoader('./data', 'train')
test_dataset = RetinopathyLoader('./data', 'test')

def showAccuracy(title='', accline=[75, 82], **kwargs):
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
    
    plt.legend(
        loc='best', bbox_to_anchor=(1.0, 1.0, 0.2, 0),
        fancybox=True, shadow=True
    )
    
    if accline:
        plt.hlines(accline, 1, len(data)+1, linestyles='dashed', colors=(0,0,0,0.8))
    
    #plt.show()
    
    return fig

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, downsampling=None):
        super(BasicBlock, self).__init__()
        padding = int(kernel_size/2)
        self.activation = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding,stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
            self.activation,
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.downsampling = downsampling
        # downsampling
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_channel != out_channel:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_channel)
        #     )

    def forward(self, x):
        input_x = x
        out = self.block(x)

        if self.downsampling is not None:
            input_x = self.downsampling(x)
        
        out += input_x
        out = self.activation(out)

        return out

class BottleneckBlock(nn.Module):
    '''
    x = (in, H, W) -> conv2d(1x1) -> conv2d -> (out, H, W) -> conv2d(1x1) -> (out*4, H, W) + x 
    '''
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, downsampling=None):
        super(BottleneckBlock, self).__init__()
        padding = int(kernel_size/2)
        self.activation = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.downsampling = downsampling
        # downsampling
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_channel != out_channel*self.expansion:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channel, out_channel*self.expansion, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(outchannel=self.expansion)
        #     )
    
    def forward(self, x):
        input_x = x
        out = self.block(x)

        if self.downsampling is not None:
            input_x = self.downsampling(x)
        
        out += input_x
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=5, start_in_channels=64):
        super(ResNet, self).__init__()

        self.current_in_channels = start_in_channels
        self.first = nn.Sequential(
            nn.Conv2d(3, self.current_in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.current_in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layers = layers
        channels = self.current_in_channels
        for i, l in enumerate(layers):
            setattr(self, 'layer'+str(i+1), 
                    self._make_layer(block, channels, l, stride=(2 if i!=0 else 1) ))
            channels*=2

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.current_in_channels, num_classes)

    def _make_layer(self, block, in_channels, blocks, stride=1):
        downsampling=None
        if stride != 1 or self.current_in_channels != in_channels * block.expansion:
            downsampling = nn.Sequential(
                nn.Conv2d(self.current_in_channels, in_channels * block.expansion, kernel_size = 1, stride=stride, bias=False),
                nn.BatchNorm2d(in_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.current_in_channels, in_channels, stride=stride, downsampling=downsampling))
        self.current_in_channels = in_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.current_in_channels, in_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        for i in range(len(self.layers)):
            x = getattr(self, 'layer'+str(i+1))(x)
        x = self.avgpool(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class PretrainResNet(nn.Module):
    def __init__(self, num_classes, num_layers):
        super(PretrainResNet, self).__init__()
        
        pretrained_model = torchvision.models.__dict__[
            'resnet{}'.format(num_layers)](pretrained=True)
        
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(
            pretrained_model._modules['fc'].in_features, num_classes
        )
                
        del pretrained_model
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def ResNet18(pre_train=False):
    if pre_train:
        return PretrainResNet(num_classes=5, num_layers=18)
    return ResNet(BasicBlock, layers=[2,2,2,2], num_classes=5)

def ResNet50(pre_train=False):
    if pre_train:
        return PretrainResNet(num_classes=5, num_layers=50)
    return ResNet(BottleneckBlock, layers=[3,4,6,3], num_classes=5)

def runModels(
    name, train_dataset, test_dataset, models, epoch_size, batch_size, learning_rate, 
    optimizer = optim.SGD, optimizer_option = {'momentum':0.9, 'weight_decay':5e-4}, 
    criterion = nn.CrossEntropyLoss(),
    show = True, testing_mode=False, callback=None
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    accs, pre_epoch = load_running_state(name, models)
    
    if accs is None:
        accs = {
            **{key+"_train" : [] for key in models},
            **{key+"_test" : [] for key in models}
        }
    if pre_epoch is None:
        pre_epoch = 0
    else:
        pre_epoch += 1
    
    if show:
        showAccuracy(
            title='Epoch [{:4d}]'.format(pre_epoch),
            **accs
        )
    
    optimizers = {
        key: optimizer(value.parameters(), lr=learning_rate, **optimizer_option) 
        for key, value in models.items()
    }
    for epoch in range(pre_epoch, epoch_size):
        bar = pyprind.ProgPercent(len(train_dataset), title="Training epoch {} : ".format(epoch+1))
        
        train_correct = {key:0.0 for key in models}
        test_correct = {key:0.0 for key in models}
        # training multiple model
        for model in models.values():
            model.train()
        
        for idx, data in enumerate(train_loader):
            x, y = data
            inputs = x.to(device)
            labels = y.to(device).long().view(-1)
        
            for optimizer in optimizers.values():
                optimizer.zero_grad()
        
            for key, model in models.items():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                cur_correct = (
                    torch.max(outputs, 1)[1] == labels
                ).sum().item()
            
                train_correct[key] += cur_correct
        
            for optimizer in optimizers.values():
                optimizer.step()
            
            bar.update(batch_size)
        
        # testing multiple model
        test_correct = evalModels(
            models, test_loader, 
            testing_mode=testing_mode
        )

        for key, value in train_correct.items():
            accs[key+"_train"] += [(value*100.0) / len(train_dataset)]
    
        for key, value in test_correct.items():
            accs[key+"_test"] += [(value*100.0) / len(test_dataset)]
         
        if show:
            clear_output(wait=True)
            showAccuracy(
                title='Epoch [{:4d}]'.format(epoch + 1),
                **accs
            )
        
        # epoch end
        torch.cuda.empty_cache()
        save_running_state(name, models, accs, epoch)
        if callback:
            callback(models, accs)
    
    return accs
Accs = {}

def evalModels(models, test_loader, testing_mode=False, return_y=False):
    test_correct = {key:0.0 for key in models}
    if return_y:
        y_pred = {key:torch.Tensor([]).long() for key in models}
        y_true = torch.Tensor([]).long()
    bar = pyprind.ProgPercent(len(test_loader.dataset), title="Testing epoch : ")
    for model in models.values():
        model.train(testing_mode)
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            x, y = data
            inputs = x.to(device)
            labels = y.to(device)
            
            if return_y:
                y_true =torch.cat((y_true, y.long().view(-1)))
            
            for key, model in models.items():
                outputs = model(inputs)
        
                test_correct[key] += (
                    torch.max(outputs, 1)[1] == labels.long().view(-1)
                ).sum().item()
                
                if return_y:
                    y_pred[key] = torch.cat((y_pred[key], torch.max(outputs, 1)[1].to(torch.device('cpu')).long()))
                
            
            bar.update(test_loader.batch_size)
            #clear_output(wait=True)
            #print('Testing batch : {:.3f} %'.format(
            #    ((idx+1)*test_loader.batch_size*100) / len(test_loader.dataset)
            #))
    if return_y:
        return test_correct, y_true, y_pred
    else:
        return test_correct

def save_running_state(name, models, accs, epoch, root='./result'):
    if not os.path.isdir(root):
        os.mkdir(root)
    root = os.path.join(root, name)
    if not os.path.isdir(root):
        os.mkdir(root)
    p = os.path.join(root, 'state.pkl')
    save_model(models, root)
    
    state = {'accs' : accs, 'epoch':epoch }
    torch.save(state, p)
        
        
def load_running_state(name, models, root='./result'):
    root = os.path.join(root, name)
    p = os.path.join(root, 'state.pkl')
    if not os.path.isfile(p):
        return None, None
    
    load_model(models, root)
    
    state = torch.load(p)
    accs = state['accs']
    epoch = state['epoch']
    return accs, epoch

def __save_model(model_name, model, root):
    if not os.path.isdir(root):
        os.mkdir(root)
    p = os.path.join(root, '{}-params.pkl'.format(model_name))
    torch.save(model.state_dict(), p)
    return p

def save_model(models, root='./model'):
    p = {}
    for k, m in models.items():
        p[k] = __save_model(k, m, root)
    return p

def __load_model(model_name, model, root):
    p = os.path.join(root, '{}-params.pkl'.format(model_name))
    if not os.path.isfile(p):
        raise AttributeError(
            "No model parameters file for {}!".format(model_name)
        )
    paras = torch.load(p)
    model.load_state_dict(paras)

def load_model(models, root='./model'):
    for k, m in models.items():
        __load_model(k, m, root)
        
def save_image(fig, title, root):
    fig.savefig(
        fname=os.path.join(root, title + '.png'),
        dpi=300,
        metadata = {
            'Title': title,
            'Author': '0756110',
        },
        bbox_inches="tight",
    )
    
def save_accuracy(name, models, accs, root='./result'):
    if not os.path.isdir(root):
        os.mkdir(root)
    root = os.path.join(root, name)
    if not os.path.isdir(root):
        os.mkdir(root)
    fig = showAccuracy(
        title=name,
        **accs
    )
    save_image(fig, title=name, root=root)
    p = os.path.join(root, name+'_accuracy.pkl')
    torch.save(accs, p)

# create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('pytorch device : ', device)
models = {
    "ResNet18" : ResNet18().to(device),
    "ResNet18(pretrain)": ResNet18(pre_train=True).to(device),
}

# Training & Testing
name = 'ResNet18'
Accs[name] = runModels(
    name, train_dataset, test_dataset, models,
    epoch_size=10, batch_size=4, learning_rate=1e-3, show=False)
fig = showAccuracy(
    title=name,
    **Accs[name]
)
fig.savefig('ResNet18.png')

# create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('pytorch device : ', device)
models = {
    "ResNet50" : ResNet50().to(device),
    "ResNet50(pretrain)": ResNet50(pre_train=True).to(device),
}

# Training & Testing
name = 'ResNet50'
Accs[name] = runModels(
    name, train_dataset, test_dataset, models,
    epoch_size=10, batch_size=4, learning_rate=1e-3, show=False
)
fig = showAccuracy(
    title=name,
    **Accs[name]
)
fig.savefig('ResNet50.png')

# create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('pytorch device : ', device)
models = {
    "ResNet18(pretrain)": ResNet18(pre_train=True).to(device),
    "ResNet50(pretrain)": ResNet50(pre_train=True).to(device),
}
# Training & Testing
name = 'ResNet Highest Accuracy 3'

def saveHighAcc(models, accs):
    for k in models.keys():
        if accs[k+'_test'][-1] >= max(accs[k+'_test']):
            __save_model(k, models[k], root='./highAccuracy')

def loadHighAcc(models):
    for k in models.keys():
        __load_model(k, models[k], root='./highAccuracy')


Accs[name] = runModels(
    name, train_dataset, test_dataset, models,
    epoch_size=10, batch_size=8, learning_rate=1e-3, show=False, callback=saveHighAcc
)
fig = showAccuracy(
    title=name,
    **Accs[name]
)

fig.savefig('best.png')

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
def plot_confusion_matrix(
    y_true, y_pred, classes,
    normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        pass

    #print(cm)

    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

loadHighAcc(models)

a, b, c = evalModels(models, test_loader = DataLoader(test_dataset, batch_size=4), return_y=True)

for k in c:
    d = float( (c[k] == b).sum() * 100 ) / len(test_dataset)
    plot_confusion_matrix(y_true=b, y_pred=c[k], title='{} {:.3f}% Confusion Matrix'.format(k, d), normalize=True, classes=np.array(['0', '1', '2', '3', '4']))


root = './result'
for n in os.listdir(root):
    a = os.path.join(root, n)
    p = os.path.join(a, 'state.pkl')
    
    if not os.path.isfile(p):
        continue
    
    state = torch.load(p)
    accs = state['accs']
    epoch = state['epoch']
    print('---')
    print(n)
    for k in accs:
        if not 'test' in k:
            continue
        print(k, ':', max(accs[k]))
    
    print('---')

    plt.show()