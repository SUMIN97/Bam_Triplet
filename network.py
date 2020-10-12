from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models

import torch.nn as nn

class StyleNet(nn.Module):
    def __init__(self):
        super(StyleNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),
            #nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 64 112 64
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            #nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 128 56 32
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
            #nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            #nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 256 28 16
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            #nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            #nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 512 14 8
            nn.Conv2d(512, 256, 3, padding=1), nn.LeakyReLU(0.2),

        )
        self.Linear = nn.Linear(256*256, 1024)
        #self.final_features = 2048

    def forward(self, img):
        # style features
        style = self.conv(img)git
        style = self.gram_matrix(style)
        # 512 * 512
        style = self.flatten(style)
        style = self.Linear(style)
        return style

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

    def flatten(self, tensor):
        tensor = torch.flatten(tensor)
        return tensor


    def pca(self, X, k):  # k is the components you want
        # mean of each feature
        mean = torch.sum(X)
        # normalization
        norm_X = X - mean
        norm_X = torch.view(1, len(X))
        # scatter matrix
        scatter_matrix = torch.dot(torch.transpose(norm_X, 0, 1), norm_X)
        # Calculate the eigenvectors and eigenvalues
        eig_val, eig_vec = np.linalg.eig(scatter_matrix.numpy())
        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(X))]
        # sort eig_vec based on eig_val from highest to lowest
        eig_pairs.sort(reverse=True)
        # select the top k eig_vec
        feature = np.array([ele[1] for ele in eig_pairs[:k]])
        # get new data
        data = np.dot(norm_X, np.transpose(feature))
        data = torch.tensor(data)
        return data



class ContentNet(nn.Module):
    def __init__(self):
        super(ContentNet, self).__init__()
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
            #nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 256 28 16
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            #nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 512 14 8
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            #nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.avg_pool = nn.AvgPool2d(7) #input이 뭐든 parameter 모양으로 만들어
        self.fc1 = nn.Linear(512 * 1 * 1, 1024)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear(1024, 1024)

    def forward(self, img):
        content = self.conv(img)
        content = self.avg_pool(content)
        content = self.flatten(content)
        content = self.fc1(content)
        content = self.relu(content)
        content = self.drop(content)
        content = self.fc2(content)

        # content = self.classifier(content)
        return content

    def flatten(self, tensor):
        tensor = torch.flatten(tensor)
        return tensor

class TripleNet(nn.Module):
    def __init__(self, style_net, content_net):
        super(TripleNet, self).__init__()
        self.style_net = style_net
        self.content_net = content_net

    def forward(self, a_img, p_img, n_img):
        anchor = self.style_net(a_img) + self.content_net(a_img)
        positive = self.style_net(p_img) + self.content_net(p_img)
        negative = self.style_net(n_img) + self.content_net(n_img)


        return anchor, positive, negative

















