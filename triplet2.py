import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.decomposition import PCA

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    #     transforms.Resize((224, 224)), #왜 224?
    transforms.ToTensor(),  # nparray => tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # 0~1값을 -0.5~0.5로 변경
])

# dataset
n_media = 7
n_content = 9
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
n_components = 512

class TripletDataset(Dataset):
    #     Train: For each sample (anchor) randomly chooses a positive and negative samples

    def __init__(self, dataset, kind):
        self.dataset = dataset
        self.kind = kind
        # self.data = []
        self.labels = []
        self.label_to_indices = {}
        self.img1 = []
        self.img2 = []
        self.img3 = []

        for m in range(n_media):
            for c in range(n_content):
                self.label_to_indices[(m, c)] = []

        for i in range(len(dataset)):
            e = self.dataset.__getitem__(i)
            m = int(e[1] / n_content)
            c = e[1] % n_content

            self.labels.append((m, c))
            self.label_to_indices[(m, c)].append(i)

        if self.kind == 'test':
            random_state = np.random.RandomState(29)
            triplets = []
            keys = list(self.label_to_indices.keys())
            anchor_key_index = -1

            for k in range(len(keys)):
                if keys[k] == self.labels[i]:
                    anchor_key_index = k

            for i in range(len(self.dataset)):
                pos_index = random_state.choice(self.label_to_indices[self.labels[i]])

                r = list(range(len(keys)))
                r.remove(anchor_key_index)
                neg_label = keys[random_state.choice(r)]
                # print(self.label_to_indices)
                neg_index = random_state.choice(self.label_to_indices[neg_label])

                triplets.append([i, pos_index, neg_index])
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.kind == 'train':
            img1, label1 = self.dataset.__getitem__(index)[0], self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])

            negative_m_range = list(range(n_media))
            negative_m_range.remove(label1[0])
            negative_m = np.random.choice(negative_m_range)

            negative_c_range = list(range(n_content))
            negative_c_range.remove(label1[1])
            negative_c = np.random.choice(negative_c_range)

            negative_label = (negative_m, negative_c)
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            img2 = self.dataset.__getitem__(positive_index)[0]
            img3 = self.dataset.__getitem__(negative_index)[0]

            self.img1 = label1
            self.img2 = self.labels[positive_index]
            self.img3 = negative_label

        elif self.kind == 'test':
            img1 = self.dataset.__getitem__(self.test_triplets[index][0])[0]
            img2 = self.dataset.__getitem__(self.test_triplets[index][1])[0]
            img3 = self.dataset.__getitem__(self.test_triplets[index][2])[0]

        else:
            print('TripletDataset Error')
            return
        # print(type(img1), type(img2), type(img3))
        return [img1, img2, img3]

    def __len__(self):
        return len(self.dataset)

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
            nn.Conv2d(512, 512, 3, padding=1)
        )
        self.fc = nn.Linear(512 * 512, 512)
        # self.fc = nn.Linear(100352, 1024)

    def gram_matrix(self, x):
        b, c, h, w = x.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = x.view(b * c, h * w)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        return G  # 512 * 512

    def gram_and_flatten(self, x):
        b, c, h, w,  = x.size()
        f = x.view(b, c, h*w)
        if cuda:
            g = torch.empty(b, c*c).cuda()
        else:
            g = torch.empty(b, c*c)
        for i in range(b):
            m = f[i]
            a = torch.mm(m, m.t())
            g[i] = a.flatten()
        return g
    

    def forward(self, x):
        output = self.conv(x)
        output = self.gram_flatten(output)
        output = output.cpu()
        output = PCA(n_components = n_components).fit_transform(output)
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
        self.fc1 = nn.Linear(512, 512)

    def forward(self, x):
        # output1 = self.convnet(x)
        print(x.size())
        output1 = self.conv(x)
        print(output1.size())
        output2 = self.avg_pool(output1)
        output3 = torch.flatten(output2)
        output4 = self.fc1(output3)

        if (math.isnan(output4[0].item())):
            print("input", x, "output1", output1, "output2", output2, "output3", output3, "output4", output4)
        return output4

train_dataset = torchvision.datasets.ImageFolder(root='./BAM', transform=trans)
test_dataset = torchvision.datasets.ImageFolder(root='./test', transform=trans)
triplet_train_dataset = TripletDataset(train_dataset, kind='train')
triplet_test_dataset = TripletDataset(test_dataset, kind='test')

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(triplet_train_dataset, batch_size=512, shuffle=True, **kwargs)
test_loader = DataLoader(triplet_test_dataset, batch_size=512, shuffle=True, **kwargs)


margin = 1
lr = 1e-4
n_epochs = 30

style_model = StyleNet()
if cuda:
    style_model.cuda()
style_parameters = style_model.parameters()
style_optimizer =  optim.Adam(style_parameters, lr=lr)

content_model = ContentNet()
if content_model.cuda():
    content_model.cuda()
content_parameters = content_model.parameters()
content_optimizer = optim.Adam(content_parameters, lr = lr)


for epoch in range(n_epochs):
    style_model.train()
    content_model.train()

    train_total_losses = 0
    for batch_idx, data in enumerate(tqdm(train_loader)):
        if cuda:
            data = [data[0].cuda(), data[1].cuda(), data[2].cuda()]

        style_optimizer.zero_grad()
        content_optimizer.zero_grad()

        print(data[0].size())
        style_outputs = [style_model(data[0]), style_model(data[1]), style_model(data[2])]
        content_outputs = [content_model(data[0]), content_model(data[1]), content_model(data[2])]

        anchor = torch.cat((style_outputs[0], content_outputs[0]), dim = 0)
        positive = torch.cat((style_outputs[1], content_outputs[1]), dim = 0)
        negative = torch.cat((style_outputs[2], content_outputs[2]), dim = 0)

        #loss
        distance_positive = (anchor - positive).pow(2).sum().sqrt()  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum().sqrt()  # .pow(.5)
        loss = F.relu(distance_positive - distance_negative + margin)
        print("loss",loss)
        for param in content_parameters:
            print("param", param.grad.view(-1))

        print("dis_pos", distance_positive, "dis_neg", distance_negative, 'loss', loss)
        train_total_losses += loss

        loss.backward()
        print()
        style_optimizer.step()
        content_optimizer.step()

    test_total_losses = 0
    with torch.no_grad():
        style_model.eval()
        content_model.eval()

        for batch_idx, data in enumerate(tqdm(test_loader)):
            if cuda:
                data = [data[0].cuda(), data[1].cuda(), data[2].cuda()]

            print(data[0].shape())
            style_outputs = [style_model(data[0]), style_model(data[1]), style_model(data[2])]
            content_outputs = [content_model(data[0]), content_model(data[1]), content_model(data[2])]

            anchor = torch.cat((style_outputs[0], content_outputs[0]), dim=0)
            positive = torch.cat((style_outputs[1], content_outputs[1]), dim=0)
            negative = torch.cat((style_outputs[2], content_outputs[2]), dim=0)

            # loss
            distance_positive = (anchor - positive).pow(2).sum().sqrt()  # .pow(.5)
            distance_negative = (anchor - negative).pow(2).sum().sqrt()  # .pow(.5)
            loss = F.relu(distance_positive - distance_negative + margin)
            test_total_losses += loss

        print("test_total_losses", test_total_losses)










