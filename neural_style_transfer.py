import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from PIL import Image
from matplotlib.pyplot import plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_ROOT = '/home/lab/Documents/PycharmProjects/bamTriplet/BAM'
TEST_ROOT = '/home/lab/Documents/PycharmProjects/bamTriplet/test'
MEDIA = ['3DGraphics', 'Comic', 'Oil', 'Pen', 'Pencil', 'VectorArt', 'Watercolor']
CONTENT = ['Bicycle', 'Bird', 'Cars', 'Cat', 'Dog', 'Flower', 'People', 'Tree']
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # torvision은 데이터셋 출력을 [0,1] 범위를  [-1, 1]의 범위로 변
    ])

NUM_RELEVANCE = 2
NUM_MEDIA = len(MEDIA)
NUM_CONTENT = len(CONTENT)

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
        while (1):
            idx = random.randint(0, len(self.loaders) -1)
            p = self.loaders[idx]
            if self.countRelevance(a, p) >= int(NUM_RELEVANCE / 2):  # relevance가 절반이상 같으면 pos image
                img = self.loaders[idx][-1]
                break;
        return img # img는 언제나 self.loader 배열에 마지막에

    def getNegImg(self, a):
        img = []
        while (1):
            idx = random.randint(0, len(self.loaders)-1)
            n = self.loaders[idx]
            if self.countRelevance(a, n) < int(NUM_RELEVANCE / 2):  # relevance가 절반이상 같으면 pos image
                img = self.loaders[idx][-1]
                break;
        return img


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
content_final_layer_ = 'conv_4'
style_final_layer = 'conv_5'

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               ):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)


    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially

    content_model = nn.Sequential(normalization)


    for layer in cnn.children():
        style_model.add_module(layer.__class__.__name__, layer)

        if layer.__class__.__name__ == style_final_layer:
            break;





