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
import h5py

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

torch.cuda.empty_cache()
n_media = 7
n_content = 9
n_components = 256
# cuda = torch.cuda.is_available()
cuda = False
device = torch.device("cuda:0" if cuda else "cpu")

folders_name = ['3DGraphics_Bicycle', '3DGraphics_Bird', '3DGraphics_Building', '3DGraphics_Cars', '3DGraphics_Cat', '3DGraphics_Dog', '3DGraphics_Flower', '3DGraphics_People', '3DGraphics_Tree', 'Comic_Bicycle', 'Comic_Bird', 'Comic_Building', 'Comic_Cars', 'Comic_Cat', 'Comic_Dog', 'Comic_Flower', 'Comic_People', 'Comic_Tree', 'Oil_Bicycle', 'Oil_Bird', 'Oil_Building', 'Oil_Cars', 'Oil_Cat', 'Oil_Dog', 'Oil_Flower', 'Oil_People', 'Oil_Tree', 'Pen_Bicycle', 'Pen_Bird', 'Pen_Building', 'Pen_Cars', 'Pen_Cat', 'Pen_Dog', 'Pen_Flower', 'Pen_People', 'Pen_Tree', 'Pencil_Bicycle', 'Pencil_Bird', 'Pencil_Building', 'Pencil_Cars', 'Pencil_Cat', 'Pencil_Dog', 'Pencil_Flower', 'Pencil_People', 'Pencil_Tree', 'VectorArt_Bicycle', 'VectorArt_Bird', 'VectorArt_Building', 'VectorArt_Cars', 'VectorArt_Cat', 'VectorArt_Dog', 'VectorArt_Flower', 'VectorArt_People', 'VectorArt_Tree', 'Watercolor_Bicycle', 'Watercolor_Bird', 'Watercolor_Building', 'Watercolor_Cars', 'Watercolor_Cat', 'Watercolor_Dog', 'Watercolor_Flower', 'Watercolor_People', 'Watercolor_Tree']

class TripletDataset(Dataset):
    #     Train: For each sample (anchor) randomly chooses a positive and negative samples

    def __init__(self, dataset, kind, ):
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
        # print(type(img1), type(img2), type(img3))
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
        self.fc = nn.Linear(512 * 512, 256)
        # self.fc = nn.Linear(100352, 1024)

    def gram_and_flatten(self, x):
        batch, c, h, w = x.size()  # a=batch size(=1)
        features = x.view(batch,c,h * w)
        G = torch.empty(batch, c*c).cuda()
        for b in range(batch):
            m = features[b]
            g = torch.mm(m, m.t())
            G[b] = g.flatten()
        return G  # (batch, 512 * 512)

    def sumin_pca(self, x):
        n_sample, n_feature = x.size() #sample, feature
        x_cen = x - torch.mean(x, 0)
        x_cov = torch.mm(x_cen.T, x_cen)/n_sample
        evalue, evector = torch.symeig(x_cov, eigenvectors = True)
        evector = torch.flip(evector, dims = [0])[:n_components]
        vector_len = torch.norm(evector, dim = 1)

        for n in range(n_components):
            evector[n]/=vector_len[n]

        component_by_sample = torch.mm(evector, x.T)
        sample_by_component = component_by_sample.T
        return sample_by_component


    def forward(self, x):
        output = self.conv(x)
        output = self.gram_and_flatten(output)
        output = self.sumin_pca(output)
        return output
        # output = output.cpu()
        # pca = PCA(n_components= n_components)
        # U, S, V = np.linalg.svd(output, full_matrices=False)
        # explained_variance_ = (S ** 2) / (n_components - 1)
        # total_var = explained_variance_.sum()
        # print(total_var)
        # print(explained_variance_)

        # output = pca.fit_transform(output)
        # print("after pca", output[0])
        # output = torch.flatten(output)
        # output = self.fc(output)

        # if cuda:
        #     output = output.cuda()
        # print('output 2 type', output.type())
        # output = self.pca(output)



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
        # output1 = self.convnet(x)
        output1 = self.conv(x)
        output2 = self.avg_pool(output1)
        output2 = output2.reshape(n_components, 512)
        output3 = self.fc1(output2)

        # if (math.isnan(output4[0].item())):
        #     print("input", x, "output1", output1, "output2", output2, "output3", output3, "output4", output4)
        return output3

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

train_dataset = torchvision.datasets.ImageFolder(root='./BAM', transform=trans)
triplet_train_dataset = TripletDataset(train_dataset, kind='train')
train_loader = DataLoader(triplet_train_dataset, batch_size=n_components, shuffle = True, **kwargs)

test_dataset = torchvision.datasets.ImageFolder(root='./test', transform=trans)
triplet_test_dataset = TripletDataset(test_dataset, kind='test')
test_loader = DataLoader(triplet_test_dataset, batch_size=(n_components), shuffle=True, **kwargs)

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


for epoch in range(n_epochs):

    style_model.train()
    content_model.train()

    train_total_losses = 0
    for batch_idx, datas in enumerate(tqdm(train_loader)):
        outputs = {
            'anchor': datas[0],
            'positive':datas[1],
            'negative':datas[2]
        }

        style_optimizer.zero_grad()
        content_optimizer.zero_grad()

        for key, data in outputs.items():
            if cuda:
                data = data.cuda()
            style = style_model(data)
            content = content_model(data)

            output = torch.cat((style.cuda(), content.cuda()), dim = 1)
            outputs[key] = output

        dis_pos = (outputs['anchor'] - outputs['positive']).pow(2).sum().pow(.5)
        dis_neg = (outputs['anchor'] - outputs['negative']).pow(2).sum().pow(.5)
        loss = F.relu(dis_pos - dis_neg + margin)
        print("loss:", loss)
        train_total_losses += loss.sum()

        loss.backward()

        is_okay_to_optimize = True
        for param in style_model.parameters():
            a =  param.grad.view(-1)[:10]
            if torch.isnan(a).sum() > 0:
                is_okay_to_optimize = False
                break;

        for param in content_model.parameters():
            a = param.grad.view(-1)[:10]
            if torch.isnan(a).sum() > 0:
                is_okay_to_optimize = False
                break;

        if is_okay_to_optimize:
            style_optimizer.step()
            content_optimizer.step()
        else:
            print("grad nan")

    print("Epoch: {}, Train Total Loss: {}".format(epoch, train_total_losses))


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


        for key, data in outputs.items():
            if cuda:
                data = data.cuda()
            style = style_model(data)
            content = content_model(data)

            output = torch.cat((style, content), dim = 1)
            outputs[key] = output

        dis_pos = (outputs['anchor'] - outputs['positive']).pow(2).sum().pow(.5)
        dis_neg = (outputs['anchor'] - outputs['negative']).pow(2).sum().pow(.5)
        loss = F.relu(dis_pos - dis_neg + margin)
        test_total_losses += loss.sum()

    print("test_total_losses", test_total_losses)











