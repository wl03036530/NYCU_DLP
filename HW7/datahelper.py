from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class CLEVRDataset(Dataset):
    def __init__(self, image_path, image_list, label_list):
        self.img_path = image_path
        self.img_list = image_list
        self.label_list = label_list
        self.transformations=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_list[index])).convert('RGB')
        img = self.transformations(img)
        return img, self.label_list[index]

if __name__ == '__main__':
    from task_1.dataset import ICLEVRLoader

    data_detail = ICLEVRLoader(os.path.join(os.getcwd(), 'task_1'))
    image_path = os.path.join(os.getcwd(), 'task_1', 'images')
    dataset = CLEVRDataset(image_path, data_detail.img_list, data_detail.label_list)

    # for idx in range(len(dataset)):
    #     data = dataset[idx]
    #     input, c = data
    #     print(input)
    #     print(c)
    
    data = dataset[0]
    input, c = data
    print(input)
    print(c)