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
        self.conv = nn.Sequential(
            # 3 224 128
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 64 112 64
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 128 56 32
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 256 28 16
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 512 14 8
            nn.Conv2d(512, 64, 3, padding=1)
        )

    def gram_and_flatten(self, x):
        batch, c, h, w = x.size()  # a=batch size(=1)
        features = x.view(batch, c, h * w)
        G = torch.empty(batch, c * c).to(device)
        for b in range(batch):
            m = features[b]
            g = torch.mm(m, m.t())
            G[b] = g.flatten()
        return G  # (batch, 512 * 512)

    def PCA_svd(self, X, k, center=True):
        n = X.size()[0]
        ones = torch.ones(n).view([n, 1])
        h = (1 / n) * torch.mm(ones, ones.t()) if center else torch.zeros(n * n).view([n, n])
        H = torch.eye(n) - h
        X_center = torch.mm(H.double().to(device), X.double())
        u, s, v = torch.svd(X_center)
        u = u[:, :k]
        x_new = u * s[:k]
        return x_new

    def forward(self, x):
        output = self.conv(x)
        output = self.gram_and_flatten(output)
        output = self.PCA_svd(output, n_components, True)

        return output


class ContentNet(nn.Module):
    def __init__(self):
        super(ContentNet, self).__init__()
        # self.convnet = nn.Sequential(*list(vgg16(pretrained=True).features))
        self.conv = nn.Sequential(
            # 3 224 128
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 64 112 64
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 128 56 32
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 256 28 16
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 512 14 8
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.avg_pool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(512, n_components)

    def forward(self, x):
        output1 = self.conv(x)
        output2 = self.avg_pool(output1)
        output2 = output2.view(-1, 512)
        output3 = self.fc1(output2)
        return output3

use_cuda = torch.cuda.is_available()
margin = 1
lr = 1e-3
n_epochs = 30
n_components = 8
batch_size = n_components

device = torch.device("cuda:1" if use_cuda else "cpu")
print("use cuda", use_cuda,"device", device)

style_model = StyleNet()
content_model = ContentNet()
if use_cuda:
    style_model.to(device)
    content_model.to(device)

style_opimizer = optim.Adam(list(style_model.parameters()), lr = lr)
content_opimizer = optim.Adam(list(content_model.parameters()), lr = lr)

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # 0~1값을 -0.5~0.5로 변경
])

for epoch in range(n_epochs):
    # Train화
    folders = glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/Grapolio_v2/유화/인물화', '*'))
    folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/Grapolio_v2/유화/동물화', '*'))
    folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/Grapolio_v2/유화/풍경화', '*'))
    folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/Grapolio_v2/유화/정물', '*'))

    dataset = TripletDataset(transform, folders)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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
        # print("positive", A.type(), B.type())
        positive = torch.cat((A, B), dim=1)
        # positive = torch.cat((style_model(positive).double(), content_model(positive).double(), dim=1)
        A = style_model(negative).double()
        B = content_model(negative).double()
        # print("negative", A.type(), B.type())
        negative = torch.cat((A, B), dim=1)

        # negative = torch.cat((style_model(negative).double(), content_model(negative)).double(), dim=1)

        dis_pos = (anchor - positive).pow(2).sum(dim=1)
        dis_neg = (anchor - negative).pow(2).sum(dim=1)
        b = dis_pos.size()[0]

        losses = F.relu(dis_pos - dis_neg + margin)
        count = (losses == torch.zeros(b).to(device)).sum()
        correct += count
        losses_sum = losses.sum()
        losses_sum.backward()
        is_okay_to_optimize = True
        for param in style_model.parameters():
            a = param.grad.view(-1)[:10]
            if torch.isnan(a).sum() > 0:
                is_okay_to_optimize = False
                break;

        for param in content_model.parameters():
            a = param.grad.view(-1)[:10]
            if torch.isnan(a).sum() > 0:
                is_okay_to_optimize = False
                break

        if is_okay_to_optimize:
            style_opimizer.step()
            content_opimizer.step()

    percent = 100. * correct / len(dataset)
    print("epoch: {}, correct: {}/{} ({:.0f}%)".format(epoch, correct, len(dataset), percent))

save_path = './model_epoch10.pth'
torch.save({
            'style_state_dict': style_model.state_dict(),
            'content_state_dict': content_model.state_dict(),
            'optimizerA_state_dict': style_opimizer.state_dict(),
            'optimizerB_state_dict': content_opimizer.state_dict(),
            }, save_path)



