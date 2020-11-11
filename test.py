import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import vgg16
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import os
from tqdm import tqdm
from glob import glob
import matplotlib
import matplotlib.pyplot as plt

torch.cuda.empty_cache()


class TripletDataset(Dataset):
    def __init__(self, imgs_path, style_transform):

        self.imgs_path = imgs_path
        self.n_imgs = len(self.imgs_path)
        self.style_transform = style_transform

    def __getitem__(self, index):

        anchor_path = self.imgs_path[index]
        anchor_material = anchor_path.split('/')[-1].split('_')[0]
        anchor_author = anchor_path.split('/')[-1].split('_')[1]

        while True:
            positive_path = self.imgs_path[np.random.randint(self.n_imgs)]
            p_material = positive_path.split('/')[-1].split('_')[0]
            p_author = positive_path.split('/')[-1].split('_')[1]
            if p_material == anchor_material and p_author == anchor_author:
                break

        while True:
            negative_path = self.imgs_path[np.random.randint(self.n_imgs)]
            n_material = negative_path.split('/')[-1].split('_')[0]
            n_author = negative_path.split('/')[-1].split('_')[1]
            if (n_material != anchor_material) and (n_author != anchor_author):
                break

        anchor_rgb = self.read_image_rgb(anchor_path) #[224, 224, 3]
        anchor_style = self.read_image_style(anchor_path) #[3, 224, 224]
        # print("anchor style size", anchor_style.size())
        # print("anchor_rgb size", anchor_rgb.size())


        positive_rgb = self.read_image_rgb(positive_path)
        positive_style = self.read_image_style(positive_path)

        negative_rgb = self.read_image_rgb(negative_path)
        negative_style = self.read_image_style(negative_path)


        return anchor_style, anchor_rgb, positive_style, positive_rgb, negative_style, negative_rgb

    def read_image_rgb(self, path):
        image = Image.open(path)
        image = image.convert('RGB')
        image = np.array(image)
        image = torch.tensor(image)
        return image

    def read_image_style(self, path):
        image = Image.open(path)
        image = image.convert('RGB')
        image = self.style_transform(image)
        return image

    def __len__(self):
        return len(self.imgs_path)


class StyleNet(nn.Module):
    def __init__(self):
        super(StyleNet, self).__init__()
        conv_list = list(vgg16(pretrained=True).features)[:-2]
        # conv_list[-1] = nn.Conv2d(512, 32, 3, padding=1)
        self.convnet = nn.Sequential(*conv_list)

    def gram_and_flatten(self, x):
        batch, c, h, w = x.size()  # a=batch size(=1)
        feature = x.view(batch, c, h * w)
        mul = torch.bmm(feature, feature.transpose(1, 2))
        return mul.view(batch, -1)  # (batch, 512 * 512)


    def forward(self, x):
        output = self.convnet(x)
        output = self.gram_and_flatten(output)
        # output = self.PCA_svd(output, n_components)
        return output



class RGB3D(nn.Module):
    def __init__(self):
        super(RGB3D, self).__init__()
        self.term = 16
        self.n = int((255 - 0 +1)/self.term +1)

    def forward(self,imgs):
        batch, height, width, color= imgs.size()

        self.r = torch.zeros(batch, self.n).to(device)
        self.g = torch.zeros(batch, self.n).to(device)
        self.b = torch.zeros(batch, self.n).to(device)


        for b in range(batch):
            for i in range(625):
                x = np.random.randint(height)
                y = np.random.randint(width)
                self.r[b][int(imgs[b][x][y][0]//self.term)] +=1
                self.g[b][int(imgs[b][x][y][1]//self.term)] +=1
                self.b[b][int(imgs[b][x][y][2] // self.term)] += 1

        self.colorspace = torch.cat((self.r, self.g, self.b), dim = 1)
        return self.colorspace






use_cuda = torch.cuda.is_available()
# use_cuda = False
margin = 0.
lr = 1e-3
n_epochs = 50
n_components = 8
batch_size = 8

device = torch.device("cuda:1" if use_cuda else "cpu")
print("use cuda", use_cuda, "device", device)

style_model = StyleNet()
rgb_model = RGB3D()


for idx, param in enumerate(style_model.parameters()):
    print("style", idx, param.size())



if use_cuda:
    style_model.to(device)
    rgb_model.to(device)


style_optimizer = optim.Adam(list(style_model.parameters()), lr=lr)


style_transform = transforms.Compose([
    # transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 0~1값을 -0.5~0.5로 변경
])




cos = nn.CosineSimilarity(dim=1, eps=1e-6)


imgs_path = glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/Grapolio_Resize_224', '*'))

dataset = TripletDataset(imgs_path, style_transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


for epoch in range(n_epochs):
    correct = 0
    for batch_idx, (anchor_style, anchor_rgb, positive_style, positive_rgb, negative_style, negative_rgb) in enumerate(tqdm(loader)):
        torch.cuda.empty_cache()

        anchor_style = anchor_style.to(device)
        anchor_rgb = anchor_rgb.to(device)

        A = style_model(anchor_style).double()
        B = rgb_model(anchor_rgb).double().to(device)


        anchor = torch.cat((A, B), dim = 1)
        positive = torch.cat((style_model(positive_style.to(device)).double(), rgb_model(positive_rgb.to(device)).double()), dim = 1)
        negative = torch.cat((style_model(negative_style.to(device)).double(), rgb_model(negative_rgb.to(device)).double()), dim = 1)


        dis_pos = 1 - cos(anchor, positive)
        dis_neg = 1 - cos(anchor, negative)

        b = dis_pos.size()[0]

        losses = torch.max(torch.zeros(b).double().to(device), dis_pos - dis_neg)


        count = (losses == torch.zeros(b).to(device)).sum()
        correct += count
        losses_sum = losses.sum()
        losses_mean = torch.mean(losses)

        style_model.zero_grad()

        losses_mean.backward()

        is_okay_to_optimize = True
        for idx, param in enumerate(style_model.parameters()):
            grad = param.grad.view(-1)
            if torch.isnan(grad).sum() > 0:
                is_okay_to_optimize = False
                print("style", idx, "grad nan")
        if is_okay_to_optimize == False:
            print(losses)
            for idx, param in enumerate(style_model.parameters()):
                print(param)

        if is_okay_to_optimize:
            style_optimizer.step()
        else:
            break

    percent = 100. * correct / len(dataset)
    print("epoch: {}, correct: {}/{} ({:.0f}%)".format(epoch, correct, len(dataset), percent))


save_path = './model_epoch50_BAM_ALL.pth'
torch.save({
            'model_state_dict': style_model.state_dict(),
            'optimizer_state_dict': style_optimizer.state_dict(),
            }, save_path)




