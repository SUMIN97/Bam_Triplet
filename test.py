import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

n_media = 7
n_content = 9
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
torch.cuda.empty_cache()
n_components = 512

class TripletDataset(Dataset):
    def __init__(self, dataset, kind):
        self.dataset = dataset
        self.kind = kind
        self.labels = {}
        self.label_to_indices = {}
        self.img1 = []
        self.img2 = []
        self.img3 = []

        for m in range(n_media):
            for c in range(n_content):
                self.label_to_indices[(m, c)] = []

        for i in range(len(dataset)):
            e = self.dataset.__getitem__(i)
            # if (e[0] == 1.0).sum() >= 7000: continue
            # elif(e[0] == -1.0).sum() >= 7000: continue
            # elif (e[0] == 0.0).sum() >= 7000: continue

            m = int(e[1] / n_content)
            c = e[1] % n_content

            self.labels[i] = ((m, c))
            self.label_to_indices[(m, c)].append(i)


        if self.kind == 'test':
            random_state = np.random.RandomState(29)
            self.test_triplets = []
            keys = list(self.label_to_indices.keys())

            for anchor_index, anchor_label in self.labels.items():
                pos_index = random_state.choice(self.label_to_indices[anchor_label])

                keys_copy = keys.copy()
                keys_copy.remove(anchor_label)
                neg_keys_index = random_state.choice(list(range(len(keys_copy))))
                neg_label = keys_copy[neg_keys_index]
                neg_index = random_state.choice(self.label_to_indices[neg_label])

                self.test_triplets.append([i, pos_index, neg_index])

    def getAvailableIndex(self):
        return list(self.labels.keys())

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

        elif self.kind == 'test':
            img1 = self.dataset.__getitem__(self.test_triplets[index][0])[0]
            img2 = self.dataset.__getitem__(self.test_triplets[index][1])[0]
            img3 = self.dataset.__getitem__(self.test_triplets[index][2])[0]

        else:
            print('TripletDataset Error')
            return
        return img1, img2, img3

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
            nn.Conv2d(512, 128, 3, padding=1)
        )
        self.fc = nn.Linear(512 * 512, 512)
        

    def gram_and_flatten(self, x):
        batch, c, h, w = x.size()
        features = x.view(batch,c,h * w)
        if cuda:
            G = torch.empty(batch, c*c).cuda()
        else:
            G = torch.empty(batch, c*c)
        for b in range(batch):
            m = features[b]
            g = torch.mm(m, m.t())
            G[b] = g.flatten()
        return G

    def pca(self, x):
        h, w = x.size()
        x_cen = x - torch.mean(x, 0)
        pca_matrix = x_cen.T
        x_conv = torch.mm(pca_matrix.T, pca_matrix)
        es, ev = torch.symeig(x_conv, eigenvectors = True)
        ev_until_n_component =  ev[:n_components, :]
        return torch.matmul(x, ev_until_n_component)

    def forward(self, x):
        output = self.conv(x)
        output = self.gram_and_flatten(output)
        output = output.cpu()
        output = PCA(n_components= n_components).fit_transform(output)
        return output


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
        output3 = torch.flatten(output2)
        output4 = self.fc1(output3)
        return output4

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

train_dataset = torchvision.datasets.ImageFolder(root='./BAM', transform=trans)
triplet_train_dataset = TripletDataset(train_dataset, kind='train')
train_loader = DataLoader(triplet_train_dataset, batch_size=n_components, shuffle = True, **kwargs)

test_dataset = torchvision.datasets.ImageFolder(root='./test', transform=trans)
triplet_test_dataset = TripletDataset(test_dataset, kind='test')
test_loader = DataLoader(triplet_test_dataset, batch_size=n_components, shuffle=True, **kwargs)

# train_sampler = SubsetRandomSampler(triplet_train_dataset.getAvailableIndex())
# train_loader = DataLoader(triplet_train_dataset, batch_size=1,sampler = train_sampler, **kwargs)





margin = 1
lr = 1e-3
n_epochs = 30

style_model = StyleNet()
if cuda:
    style_model.cuda()
style_parameters = style_model.parameters()
style_optimizer =  optim.Adam(style_parameters, lr=lr)

content_model = ContentNet()
if cuda:
    content_model.cuda()
content_parameters = content_model.parameters()
content_optimizer = optim.Adam(content_parameters, lr = lr)

"""
for epoch in range(n_epochs):

    style_model.train()
    content_model.train()

    train_total_losses = 0
    for batch_idx, data in enumerate(tqdm(train_loader)):
        # if batch_idx == 100: break;

        if cuda:
            data = [data[0].cuda(), data[1].cuda(), data[2].cuda()]

        style_optimizer.zero_grad()
        content_optimizer.zero_grad()


        style_outputs = [style_model(data[0]), style_model(data[1]), style_model(data[2])]
        content_outputs = [content_model(data[0]), content_model(data[1]), content_model(data[2])]

        anchor = torch.cat((style_outputs[0], content_outputs[0]), dim = 0)
        positive = torch.cat((style_outputs[1], content_outputs[1]), dim = 0)
        negative = torch.cat((style_outputs[2], content_outputs[2]), dim = 0)


        #loss
        distance_positive = (anchor - positive).pow(2).sum().sqrt()  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum().sqrt()  # .pow(.5)
        loss = F.relu(distance_positive - distance_negative + margin)

        # print("loss",loss)
        # if math.isnan(loss):
        #     print("data[0]", data[0], "data[1]", data[1], "data[2]", data[2])
        #     print("anchor", anchor, "positive", positive, "negative", negative)
        #     for param in style_model.parameters():
        #         print("style param", param.view(-1))
        #     for param in content_model.parameters():
        #         print("content param", param.view(-1))
        #     continue

        train_total_losses += loss

        loss.backward()

        is_okay_to_optimize = True
        for param in style_model.parameters():
            a =  param.grad.view(-1)[:10]
            # print("style grad",a)
            if torch.isnan(a).sum() > 0:
                is_okay_to_optimize = False
                break;


        for param in content_model.parameters():
            a = param.grad.view(-1)[:10]
            # print("content grad", a)
            if torch.isnan(a).sum() > 0:
                is_okay_to_optimize = False
                break;

        if is_okay_to_optimize:
            style_optimizer.step()
            content_optimizer.step()
        else:
            print("grad nan")
"""

test_total_losses = 0
with torch.no_grad():
    # style_model.eval()
    # content_model.eval()

    for batch_idx, datas in enumerate(tqdm(test_loader)):
        outputs = {
            'anchor': datas[0],
            'positive':datas[1],
            'negative':datas[2]
        }

        print(datas[0].size())
        for key, data in outputs.items():
            if cuda:
                data = data.cuda()

            output = torch.cat((style_model(data), content_model(data)), dim = 0)
            outputs[key] = output

        dis_pos = (outputs['anchor'] - outputs['positive']).pow(2).sum().pow(.5)
        dis_neg = (outputs['anchor'] - outputs['negative']).pow(2).sum().pow(.5)
        loss = F.relu(dis_pos - dis_neg + margin)
        test_total_losses += loss.sum()

    print("test_total_losses", test_total_losses)
