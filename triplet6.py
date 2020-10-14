import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.decomposition import PCA
import h5py
from glob import glob
from torch.autograd import Variable


torch.cuda.empty_cache()

class TripletDataset(Dataset):
    def __init__(self, folder_path, transform, folders):
        self.transform = transform
        self.folder_path = folder_path
    
        self.img_list = os.listdir(self.folder_path)
        self.folders = folders


    def __getitem__(self, index):
        anchor = self.read_image(os.path.join(self.folder_path, self.img_list[index]))

        positive_index = index
        while positive_index == index:
            positive_index = np.random.randint(len(self.img_list))
        posive = self.read_image(os.path.join(self.folder_path, self.img_list[positive_index]))

        negative_path = self.folder_path
        while negative_path == self.folder_path:
            negative_path = self.folders[np.random.randint(len(self.folders))]
        
        negative_list = os.listdir(negative_path)
        negative_index = np.random.randint(len(negative_list))
        negative = self.read_image(os.path.join(negative_path, negative_list[negative_index]))

        return anchor, posive, negative


    def read_image(self, path):
        image = Image.open(path)
        image = image.convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_list)



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
        evalue, evector = torch.symeig(x_cov, eigenvectors = True) #오름차순
        evector = torch.flip(evector, dims = [0])[:n_components] #내림차순 & N_COMPONENT => (n_components, feature)
        vector_len = torch.norm(evector, dim = 1)

        for n in range(n_components):
            evector[n]/=vector_len[n]

        component_by_sample = torch.mm(evector, x.T)
        sample_by_component = component_by_sample.T
        return sample_by_component

    def PCA_eig(self, X,k, center=True, scale=False):
        n,p = X.size()
        print(X)
        ones = torch.ones(n).view([n,1])
        h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
        H = torch.eye(n) - h
        X_center =  torch.mm(H.double(), X.double())
        print("X_center", X_center)
        covariance = 1/(n-1) * torch.mm(X_center.t(), X_center).view(p,p)
        print("convariance", covariance)
        scaling =  torch.sqrt(1/torch.diag(covariance)).double() if scale else torch.ones(p).double()
        print("scaling", scaling)
        A = torch.diag(scaling).view(p,p)
        print("A", A)
        print("A size", A.size(), "covariance size", covariance.size())
        scaled_covariance = torch.empty(p,p)
        scaled_covariance = torch.mm(A, covariance)
        
        print(scaled_covariance)
        eigenvalues, eigenvectors = torch.eig(scaled_covariance, True)
        components = (eigenvectors[:, :k]).t()
        projection = torch.mm(X, components.t())
        return projection
    
    def PCA_svd(self, X, k, center=True):
        n = X.size()[0]

        X_center = Variable(torch.empty(n, X.size()[1]), requires_grad=True)
        ones = Variable(torch.ones(n).view([n,1]), requires_grad=True)
        h = Variable((1/n) * torch.mm(ones, ones.t()), requires_grad=True) if center  else torch.zeros(n*n).view([n,n])
        H = Variable(torch.eye(n) - h, requires_grad= True)
        X_center =  Variable(torch.mm(H.double().cuda(), X.double()), requires_grad=True)
        u, s, v = torch.svd(X_center)
        # u = Variable(u, requires_grad=True)
        # s = Variable(u, requires_grad=True)
        # v = Variable(u, requires_grad=True)
        u *= s[:k]
        u = Variable(u, requires_grad=True)
        return u


    def forward(self, x):
        output = self.conv(x)
        output = self.gram_and_flatten(output)
        output = self.PCA_svd(output, n_components, True)
        return output
    """
    def forward(self, x):
        output = self.conv(x)
        output = self.gram_and_flatten(output)
        output = output.cpu()
        print(output.type())
        pca = PCA(n_components=n_components)
        output = pca.fit_transform(output)
        output = torch.from_numpy(output)
        return output
    """


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
        output2 = output2.view(-1, 512)
        output3 = self.fc1(output2)
        # if (math.isnan(output4[0].item())):
        #     print("input", x, "output1", output1, "output2", output2, "output3", output3, "output4", output4)
        return output3

use_cuda = torch.cuda.is_available()
#kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
margin = 1
lr = 1e-4
n_epochs = 30
n_components = 32

# cuda = False
device = torch.device("cuda:0" if use_cuda else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
print("use cuda", use_cuda,"device", device)

style_model = StyleNet()
style_mocel = nn.DataParallel(style_model)
content_model = ContentNet()
content_model = nn.DataParallel(content_model)

if use_cuda:
    style_model.cuda()
    content_model.cuda()

style_opimizer = optim.Adam(list(style_model.parameters()), lr = lr)
content_opimizer = optim.Adam(list(content_model.parameters()), lr = lr)

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # 0~1값을 -0.5~0.5로 변경
])


for epoch in range(n_epochs):
    print("epoch", epoch)
    #Train
    folders = glob(os.path.join('/home/lab/Documents/SWMaestro/AWS/Bam_Triplet/BAM', '*'))
    # folders = glob(os.path.join('/home/ubuntu/artri/Bam_Triplet/BAM', '*'))
    
    for folder in folders:
        dataset = TripletDataset(folder, transform, folders)
        loader = DataLoader(dataset, batch_size = n_components, shuffle= True)

        train_loss = 0.0
        for batch_idx, (anchor, positive, negative) in enumerate(tqdm(loader)):
            torch.cuda.empty_cache()
            if use_cuda:
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

            anchor = torch.cat((style_model(anchor).double(), content_model(anchor).double()), dim = 1)
            A = style_model(positive).double()
            B = content_model(positive).double()
            print("positive", A.type(), B.type())
            positive = torch.cat((A, B), dim=1)
            # positive = torch.cat((style_model(positive).double(), content_model(positive).double(), dim=1)
            A = style_model(negative).double()
            B = content_model(negative).double()
            print("negative", A.type(), B.type())
            negative = torch.cat((A, B), dim=1)

            # negative = torch.cat((style_model(negative).double(), content_model(negative)).double(), dim=1)

            dis_pos = (anchor - positive).pow(2).sum().pow(.5)
            dis_neg = (anchor - negative).pow(2).sum().pow(.5)
            loss = torch.max(dis_pos - dis_neg + margin, torch.tensor([0.0], dtype = torch.double).cuda())


            train_loss += loss.item()

            style_opimizer.zero_grad()
            content_opimizer.zero_grad()
            loss.backward()

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
        print("train_loss", train_loss)




