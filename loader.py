import matplotlib.pyplot as plt
import numpy as np
import random
import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torchvision
from torchvision import transforms, models

from network import *
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
LEARNING_RATE = 0.0001


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

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataLoader = Loader(TRAIN_ROOT)
data = dataLoader.forward()
tripleNet = TripleNet(StyleNet(), ContentNet())
tripleNet = nn.DataParallel(tripleNet)
tripleNet = tripleNet.cuda()

parameters = tripleNet.parameters()
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

        anchor, positive, negative = tripleNet.forward(a_img, p_img, n_img)

        loss = loss_fn.forward(anchor,positive, negative)

        if idx % 100 == 0:
            print("loss", loss)
        if loss != 0:
            loss.backward()

        optimizer.step()


# #test
#
# testLoader = Loader(TEST_ROOT)
# test = testLoader.forward()
#
# for idx in range(len(test)):





# mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
#               '#bcbd22', '#17becf']
#
# def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
#     plt.figure(figsize=(10,10))
#     for i in range(10):
#         inds = np.where(targets==i)[0]
#         plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
#     if xlim:
#         plt.xlim(xlim[0], xlim[1])
#     if ylim:
#         plt.ylim(ylim[0], ylim[1])
#     plt.legend(mnist_classes)
#
# def extract_embeddings(dataloader, model):
#     with torch.no_grad():
#         model.eval()
#         embeddings = np.zeros((len(dataloader.dataset), 2))
#         labels = np.zeros(len(dataloader.dataset))
#         k = 0
#         for images, target in dataloader:
#             if cuda:
#                 images = images.cuda()
#             embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
#             labels[k:k+len(images)] = target.numpy()
#             k += len(images)
#     return embeddings, labelsvv






















