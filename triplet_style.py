import matplotlib.pyplot as plt
import numpy as np
import random
import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
from torchvision import transforms, models

from losses import TripletLoss


TRAIN_ROOT = '/home/lab/Documents/PycharmProjects/bamTriplet/BAM'
TEST_ROOT = '/home/lab/Documents/PycharmProjects/bamTriplet/test'
MEDIA = ['3DGraphics', 'Comic', 'Oil', 'Pen', 'Pencil', 'VectorArt', 'Watercolor']
CONTENT = ['Bicycle', 'Bird', 'Cars', 'Cat', 'Dog', 'Flower', 'People', 'Tree']
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # torvision은 데이터셋 출력을 [0,1] 범위를  [-1, 1]의 범위로 변

EPOCH = 20
NUM_MEDIA = len(MEDIA)
NUM_CONTENT = len(CONTENT)
NUM_RELEVANCE = 2
MARGIN = 0.6
LAMDA = 5 * (10**4)
LEARNING_RATE = 0.001


class Loader():
    def __init__(self, TRAIN_ROOT):
        self.loaders = []

    def forward(self):
        for m in range(NUM_MEDIA):
            dataset = torchvision.datasets.ImageFolder(root=os.path.join(TRAIN_ROOT, MEDIA[m]), transform=TRANSFORM)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
            for i, data in enumerate(dataloader):
                img = data[0].cuda()
                c = data[1].item()
                element = [m, c, img]
                self.loaders.append(element)

        return self.loaders

    def countRelevance(self, input1, input2):
        n = 0
        for i in range(NUM_RELEVANCE):
            if input1[i] == input2[i]:
                n += 1
        return n

    def getPosImg(self, a):
        img = []
        idx = 0
        while (1):
            idx = random.randint(0, len(self.loaders) -1)
            p = self.loaders[idx]
            if self.countRelevance(a, p) >= int(NUM_RELEVANCE / 2):  # relevance가 절반이상 같으면 pos image
                img = self.loaders[idx][-1]
                break;
        # print("pos",idx)
        return img # img는 언제나 self.loader 배열에 마지막에

    def getNegImg(self, a):
        img = []
        idx = 0
        while (1):
            idx = random.randint(0, len(self.loaders)-1)
            n = self.loaders[idx]
            if self.countRelevance(a, n) < int(NUM_RELEVANCE / 2):  # relevance가 절반이상 같으면 pos image
                img = self.loaders[idx][-1]
                break;
        # print("neg", idx)
        return img

class StyleNet(nn.Module):
    def __init__(self):
        super(StyleNet, self).__init__()
        self.conv = nn.Sequential(
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
            nn.Conv2d(512, 256, 3, padding=1), nn.LeakyReLU(0.2),

        )



    def forward(self, img):
        # style features
        style = self.conv(img)
        style = self.gram_matrix(style)
        style = self.flatten(style)
        return style

    def flatten(self, tensor):
        tensor = torch.flatten(tensor)
        return tensor

    # def gram_matrix(self, tensor):
    #
    #     # get the batch_size, depth, height, and width of the Tensor
    #     b, c, h, w = tensor.size()
    #     # reshape so we're multiplying the features for each channel
    #     tensor = tensor.view(b, c, h * w)
    #     # calculate the gram matrix
    #     gram = tensor.bmm(tensor.transpose(1, 2)) / (c * h * w)
    #     return gram

    def gram_matrix(self,input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

dataLoader = Loader(TRAIN_ROOT)
data = dataLoader.forward()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
styleNet = StyleNet()
styleNet.cuda(device)

parameters = styleNet.parameters()
optimizer = optim.Adam(parameters, lr=LEARNING_RATE, weight_decay = 10**(-6))
optimizer.zero_grad()

loss_fn = TripletLoss(MARGIN)

numData = len(data)
print(numData)

for epoch in range((EPOCH)):
    print("#################EPOCH %d ###########"%epoch)
    for idx in tqdm(range(numData)):
        a = data[idx]
        a_img = a[-1]
        p_img = dataLoader.getPosImg(a)
        n_img = dataLoader.getNegImg(a)

        anchor = styleNet.forward(a_img)
        positive = styleNet.forward(p_img)
        negative = styleNet.forward(n_img)

        loss = loss_fn.forward(anchor, positive, negative)

        if idx % 100 == 0:
            print("loss", loss)
        if loss != 0:
            loss.backward()

        optimizer.step()

























