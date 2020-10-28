import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision.models import vgg16
import torch.optim as optim
from torchvision import transforms
from glob import glob
import os


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
        output = self.PCA_svd(output, n_components)
        return output

class ContentNet(nn.Module):
    def __init__(self):
        super(ContentNet, self).__init__()
        self.convnet = nn.Sequential(*list(vgg16(pretrained=True).features))
        self.avg_pool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(512, n_components)

    def forward(self, x):
        output1 = self.convnet(x)
        output2 = self.avg_pool(output1)
        output2 = output2.view(-1, 512)
        output3 = self.fc1(output2)
        return output3

img_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 0~1값을 -0.5~0.5로 변경
])

use_cuda = torch.cuda.is_available()
# use_cuda = False
margin = 0.
lr = 1e-3
n_epochs = 50
n_components = 6
batch_size = 6

device = torch.device("cuda:0" if use_cuda else "cpu")
print("use cuda", use_cuda, "device", device)

folders = glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/test/3DGraphics_Bicycle', '*'))
folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/test/3DGraphics_Dog', '*'))
folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/test/Watercolor_Bicycle', '*'))
folders += glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/test/Watercolor_Dog', '*'))

style_model = StyleNet()
content_model = ContentNet()
style_optimizer = optim.Adam(list(style_model.parameters()), lr=lr)
content_optimizer = optim.Adam(list(content_model.parameters()), lr=lr)

checkpoint = torch.load("model_epoch50_v2.pth")
style_model.load_state_dict(checkpoint['style_state_dict'])
content_model.load_state_dict(checkpoint['content_state_dict'])
style_optimizer.load_state_dict(checkpoint['optimizerA_state_dict'])
content_optimizer.load_state_dict(checkpoint['optimizerB_state_dict'])

style_model.eval()
content_model.eval()

anchor_path = '/home/lab/Documents/ssd/SWMaestro/test/3DGraphics_Bicycle/3DGraphics_Bicycle_0.jpg'
anchor = Image.open(anchor_path)
anchor = anchor.convert('RGB')
anchor = img_transform(anchor)



