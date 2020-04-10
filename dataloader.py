from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import os
import torchvision
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list
        self.n_class = len(list(set(target_list)))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

    def __len__(self):
        self.len = len(self.file_list)
        return self.len*2

    def __getitem__(self, index):
        img = Image.open(self.file_list[index%self.len])

        if (index / self.len == 1):
            img = self.transform2(img)
        # elif (index / self.len == 2):
        #     img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        label = self.target_list[index%self.len]
        return img, label

class testImageDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        return img

class verfImageDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img1 = Image.open(self.file_list[index][0])
        img2 = Image.open(self.file_list[index][1])

        img1 = torchvision.transforms.ToTensor()(img1)
        img2 = torchvision.transforms.ToTensor()(img2)
        return img1, img2

def parse_data(datadir,load_mode, target_dict = {}):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):  #root: median/1
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(root.split('/')[-1])

    # construct a dictionary, where key and value correspond to ID and target
    uniqueID_list = list(set(ID_list))
    class_n = len(uniqueID_list)
    if load_mode == 'train':
        target_dict = dict(zip(uniqueID_list, range(class_n)))
        label_dict = dict(zip(range(class_n), uniqueID_list))
        label_list = [target_dict[ID_key] for ID_key in ID_list]
        return img_list, label_list, class_n, target_dict, label_dict
    else:
        label_list = [target_dict[ID_key] for ID_key in ID_list]

    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
    return img_list, label_list, class_n

def load_data_train(datadir, batch_size):
    img_list, label_list, class_n, target_dict, label_dict = parse_data(datadir,'train')
    trainset = ImageDataset(img_list, label_list)
    dataloader = DataLoader(trainset, batch_size = batch_size, shuffle=True, num_workers=1, drop_last=False)
    return dataloader, class_n, len(img_list), target_dict, label_dict

def load_data_val(datadir, batch_size, target_dict):
    img_list, label_list, class_n = parse_data(datadir, 'val', target_dict)
    trainset = ImageDataset(img_list, label_list)
    dataloader = DataLoader(trainset, batch_size = batch_size, shuffle=True, num_workers=1, drop_last=False)
    return dataloader, class_n, len(img_list)


def load_data_vision(datadir, batch_size):
    imageFolder_dataset = torchvision.datasets.ImageFolder(root=datadir,transform=torchvision.transforms.ToTensor())
    imageFolder_dataloader = DataLoader(imageFolder_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    return imageFolder_dataloader, len(imageFolder_dataset.classes), imageFolder_dataset.__len__()