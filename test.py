import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from torchvision.models import vgg16
from torch import autograd

torch.cuda.empty_cache()


class TripletDataset(Dataset):
    def __init__(self, transform, imgs_path):
        self.transform = transform
        self.imgs_path = imgs_path
        self.n_imgs = len(self.imgs_path)

    def __getitem__(self, index):

        anchor_path = self.imgs_path[index]
        anchor_material = anchor_path.split('/')[-1].split('_')[0]
        anchor_author = anchor_path.split('/')[-1].split('_')[1]

        while True:
            positive_path = self.imgs_path[np.random.randint(self.n_imgs)]
            p_material = positive_path.split('/')[-1].split('_')[0]
            p_author = positive_path.split('/')[-1].split('_')[1]
            if p_material == anchor_material or p_author == anchor_author:
                break

        while True:
            negative_path = self.imgs_path[np.random.randint(self.n_imgs)]
            n_material = negative_path.split('/')[-1].split('_')[0]
            n_author = negative_path.split('/')[-1].split('_')[1]
            if (n_material != anchor_material) and (n_author != anchor_author):
                break

        anchor = self.read_image(anchor_path)
        positive = self.read_image(positive_path)
        negative = self.read_image(negative_path)

        return anchor, positive, negative

    def read_image(self, path):
        image = Image.open(path)
        image = image.convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.imgs_path)


class StyleNet(nn.Module):
    def __init__(self):
        super(StyleNet, self).__init__()
        #         self.conv = nn.Sequential(
        #             # 3 224 128
        #             nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.MaxPool2d(2, 2),
        #             # 64 112 64
        #             nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.MaxPool2d(2, 2),
        #             # 128 56 32
        #             nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.MaxPool2d(2, 2),
        #             # 256 28 16
        #             nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.MaxPool2d(2, 2),
        #             # 512 14 8
        #             nn.Conv2d(512, 64, 3, padding=1)
        #         )
        conv_list = list(vgg16(pretrained=True).features)[:-2]
        conv_list[-1] = nn.Conv2d(512, 32, 3, padding=1)
        self.convnet = nn.Sequential(*conv_list)



    def gram_and_flatten(self, x):
        batch, c, h, w = x.size()  # a=batch size(=1)
        feature = x.view(batch, c, h * w)
        mul = torch.bmm(feature, feature.transpose(1, 2))
        return mul.view(batch, -1)  # (batch, 512 * 512)

    def sumin_pca(self, X, k):

        u, s, v = torch.pca_lowrank(X, center=True)
        return torch.mm(X, v[:, :k])

    def PCA_svd(self, X, k, center=True):
        n = X.size()[0]
        ones = torch.ones(n).view([n, 1])
        h = (1 / n) * torch.mm(ones, ones.t()) if center else torch.zeros(n * n).view([n, n])
        H = torch.eye(n) - h
        X_center = torch.mm(H.double().to(device), X.double())

        try:
            u, s, v = torch.svd(X_center)
        except:

            u, s, v = torch.svd(X_center + 1e-4 * X_center.mean())
        x_new = torch.mm(X, v[:, :k].float())
        return x_new

    def forward(self, x):
        output = self.convnet(x)
        output = self.gram_and_flatten(output)
        # print(output.size())
        # output = self.sumin_pca(output, n_components)
        output = self.PCA_svd(output, n_components)
        return output


class ContentNet(nn.Module):
    def __init__(self):
        super(ContentNet, self).__init__()
        self.convnet = nn.Sequential(*list(vgg16(pretrained=True).features))
        #         self.conv = nn.Sequential(
        #             # 3 224 128
        #             nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.MaxPool2d(2, 2),
        #             # 64 112 64
        #             nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.MaxPool2d(2, 2),
        #             # 128 56 32
        #             nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.MaxPool2d(2, 2),
        #             # 256 28 16
        #             nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.MaxPool2d(2, 2),
        #             # 512 14 8
        #             nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
        #             nn.MaxPool2d(2, 2)
        #         )
        self.avg_pool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(512, n_components)

    def forward(self, x):
        output1 = self.convnet(x)
        output2 = self.avg_pool(output1)
        output2 = output2.view(-1, 512)
        output3 = self.fc1(output2)
        return output3


use_cuda = torch.cuda.is_available()
# use_cuda = False
margin = 0.
lr = 1e-3
n_epochs = 50
n_components = 6
batch_size = 6

device = torch.device("cuda:0" if use_cuda else "cpu")
print("use cuda", use_cuda, "device", device)

style_model = StyleNet()
content_model = ContentNet()

# print(style_model)
# print(content_model)

for idx, param in enumerate(style_model.parameters()):
    print("style", idx, param.size())
#
#
# for idx, param in enumerate(content_model.parameters()):
#     print("content", idx, param.size())


if use_cuda:
    style_model.to(device)
    content_model.to(device)

style_optimizer = optim.Adam(list(style_model.parameters()), lr=lr)
content_optimizer = optim.Adam(list(content_model.parameters()), lr=lr)

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 0~1값을 -0.5~0.5로 변경
])

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

folders = glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/test/3DGraphics_Bicycle', '*'))
folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/test/3DGraphics_Dog', '*'))
folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/test/Watercolor_Bicycle', '*'))
folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/test/Watercolor_Dog', '*'))
dataset = TripletDataset(transform, folders)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


for epoch in range(n_epochs):


    # Train화
    # folders = glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/Grapolio_v2/Resize_224/oil/인물화', '*'))
    # folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/Grapolio_v2/Resize_224/oil/동물화', '*'))
    # folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/Grapolio_v2/Resize_224/oil/풍경화', '*'))
    # folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/Grapolio_v2/Resize_224/oil/정물화', '*'))
    # folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/Grapolio_v2/Resize_224/oil/추상화', '*'))

    correct = 0
    for batch_idx, (anchor, positive, negative) in enumerate(tqdm(loader)):
        torch.cuda.empty_cache()

        if use_cuda:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

        anchor = torch.cat((style_model(anchor).double(), content_model(anchor).double()), dim=1)
        A = style_model(positive).double()
        B = content_model(positive).double()

        positive = torch.cat((A, B), dim=1)
        # positive = torch.cat((style_model(positive).double(), content_model(positive).double(), dim=1)
        A = style_model(negative).double()
        B = content_model(negative).double()

        negative = torch.cat((A, B), dim=1)

        # negative = torch.cat((style_model(negative).double(), content_model(negative)).double(), dim=1)

        dis_pos = 1 - cos(anchor, positive)
        dis_neg = 1 - cos(anchor, negative)

        #         dis_pos = (anchor - positive).pow(2).sum(dim=1)
        #         dis_neg = (anchor - negative).pow(2).sum(dim=1)
        b = dis_pos.size()[0]

        # losses = F.relu(dis_pos - dis_neg)
        losses = torch.max(torch.zeros(b).double().to(device), dis_pos - dis_neg)


        count = (losses == torch.zeros(b).to(device)).sum()
        correct += count
        # losses_sum = losses.sum()
        losses_mean = torch.mean(losses)
        # print(losses_mean.item())


        # style_optimizer.zero_grad()
        # content_optimizer.zero_grad()
        style_model.zero_grad()
        content_model.zero_grad()


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

        for idx, param in enumerate(content_model.parameters()):
            grad = param.grad.view(-1)
            # print("content", idx, grad[:5])
            if torch.isnan(grad).sum() > 0:
                is_okay_to_optimize = False
                print("content", idx, "grad nan")
                break

        if is_okay_to_optimize:
            style_optimizer.step()
            content_optimizer.step()
        else:
            break

    percent = 100. * correct / len(dataset)
    print("epoch: {}, correct: {}/{} ({:.0f}%)".format(epoch, correct, len(dataset), percent))

save_path = './model_epoch50_v2.pth'
torch.save({
            'style_state_dict': style_model.state_dict(),
            'content_state_dict': content_model.state_dict(),
            'optimizerA_state_dict': style_optimizer.state_dict(),
            'optimizerB_state_dict': content_optimizer.state_dict(),
            }, save_path)





