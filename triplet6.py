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


import matplotlib
import matplotlib.pyplot as plt


torch.cuda.empty_cache()

class TripletDataset(Dataset):
    def __init__(self, folder_path, transform, folders):
        self.transform = transform
        self.folder_path = folder_path
        self.style = self.folder_path.split('/')[-1].split('_')[0]
        self.content = self.folder_path.split('/')[-1].split('_')[1]
        self.img_list = os.listdir(self.folder_path)
        self.folders = folders


    def __getitem__(self, index):
        anchor = self.read_image(os.path.join(self.folder_path, self.img_list[index]))

        while True:
            positive_path = self.folders[np.random.randint(len(self.folders))]
            p_style = positive_path.split('/')[-1].split('_')[0]
            p_content = positive_path.split('/')[-1].split('_')[1]
            if p_style == self.style or p_content == self.content: break

        positive_list = os.listdir(positive_path)
        positive_index = np.random.randint(len(positive_list))
        positive = self.read_image(os.path.join(positive_path, positive_list[positive_index]))

        negative_path = self.folder_path
        while negative_path == self.folder_path:
            negative_path = self.folders[np.random.randint(len(self.folders))]
        
        negative_list = os.listdir(negative_path)
        negative_index = np.random.randint(len(negative_list))
        negative = self.read_image(os.path.join(negative_path, negative_list[negative_index]))

        return anchor, positive, negative


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
        ones = torch.ones(n).view([n,1])
        # ones = Variable(ones, requires_grad=True)
        h = (1/n) * torch.mm(ones, ones.t()) if center  else torch.zeros(n*n).view([n,n])
        # h = Variable(h, requires_grad=True)
        H = torch.eye(n) - h
        # H= Variable(H, requires_grad=True)
        X_center =  torch.mm(H.double().cuda(), X.double())
        # X_center = Variable(X_center, requires_grad=True)
        u, s, v = torch.svd(X_center)
        u = u[:, :k]
        x_new = u * s[:k]
        # s = Variable(s, requires_grad=True)
        # u = Variable(u, requires_grad=True)
        return x_new


    def forward(self, x):
        output = self.conv(x)
        output = self.gram_and_flatten(output)
        output = self.PCA_svd(output, n_components, True)
        # output = self.PCA_svd(output, 1, True) #좌
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
        # self.fc1 = nn.Linear(512, 1) #y좌표

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

n_components = 16#32
batch_size = n_components


# cuda = False
device = torch.device("cuda:1" if use_cuda else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
print("use cuda", use_cuda,"device", device)

style_model = StyleNet()
# style_mocel = nn.DataParallel(style_model)
content_model = ContentNet()
# content_model = nn.DataParallel(content_model)

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
    #Train
    folders = sorted(glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/BAM', '*')))
    # folders = glob(os.path.join('/home/ubuntu/artri/Bam_Triplet/BAM', '*'))
    total_correct = 0

    for folder in folders:
        dataset = TripletDataset(folder, transform, folders)
        loader = DataLoader(dataset, batch_size = batch_size, shuffle= True)

        correct = 0
        for batch_idx, (anchor, positive, negative) in enumerate(tqdm(loader)):
            torch.cuda.empty_cache()
            if use_cuda:
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

            anchor = torch.cat((style_model(anchor).double(), content_model(anchor).double()), dim = 1)
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


            dis_pos = (anchor - positive).pow(2).sum(dim = 1).pow(.5)
            dis_neg = (anchor - negative).pow(2).sum(dim = 1).pow(.5)
            b = dis_pos.size()[0]

            losses = F.relu(dis_pos - dis_neg + margin)
            count = (losses == torch.zeros(b).cuda()).sum()
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
        print("correct: {}/{} ({:.0f}%)".format(correct, len(dataset), percent))
        total_correct += correct
    print("epoch", epoch, 'total_correct', total_correct)

save_path = './model_epoch30.pth'
torch.save({
            'style_state_dict': style_model.state_dict(),
            'content_state_dict': content_model.state_dict(),
            'optimizerA_state_dict': style_opimizer.state_dict(),
            'optimizerB_state_dict': content_opimizer.state_dict(),
            }, save_path)

# https://html-color-codes.info/    content 가 같으면 같은색
# Bicycle : 빨
# bird : 주
# building :노
#cars: 초
# cat: 민트
# dog: 파
# flower: 보
# people:분홍
# tree : 회
#
# 명도: (흰) 3dgraphic > comic > oil > pen > pencil > vectorart > watercolor




#9로 나눈 몫
alpha_dict = {
    0: 0.2,
    1: 0.3,
    2: 0.4,
    3:0.6,
    4:0.7,
    5:0.8,
    6:1.0
}

#9로 나눈 나머
color_dict = {
    0: np.array([255, 0, 0]),
    1: np.array([255, 112, 0]),
    2:  np.array([255, 255, 0]),
    3: np.array([0, 255, 0]),
    4: np.array([0, 255, 199]),
    5: np.array([0, 0, 255]),
    6: np.array([138, 0, 255]),
    7:np.array([249, 0, 255]),
    8:np.array([21, 22, 14])
}

plt.figure(figsize=(20,20))

print("\n\nThis is test:\n\n")
with torch.no_grad():
    style_model.eval()
    content_model.eval()

    folders = glob(os.path.join('/home/lab/Documents/ssd/SWMaestro/test', '*'))
    for idx, folder in enumerate(folders):
        dataset = TripletDataset(folder, transform, folders)
        loader = DataLoader(dataset, batch_size = batch_size, shuffle= True)

        correct = 0
        for batch_idx, (anchor, positive, negative) in enumerate(tqdm(loader)):
            torch.cuda.empty_cache()
            if use_cuda:
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

            anchor = torch.cat((style_model(anchor).double(), content_model(anchor).double()), dim = 1)
            A = style_model(positive).double()
            B = content_model(positive).double()
            # print("positive", A.type(), B.type())
            positive = torch.cat((A, B), dim=1)
            # positive = torch.cat((style_model(positive).double(), content_model(positive).double(), dim=1)
            A = style_model(negative).double()
            B = content_model(negative).double()
            # print("negative", A.type(), B.type())
            negative = torch.cat((A, B), dim=1)

            dis_pos = (anchor - positive).pow(2).sum(dim = 1).pow(.5)
            dis_neg = (anchor - negative).pow(2).sum(dim = 1).pow(.5)
            b = dis_pos.size()[0]

            losses = F.relu(dis_pos - dis_neg + margin)
            count = (losses == torch.zeros(b).cuda()).sum()
            correct += count

        percent = 100. * correct / len(dataset)
        print("correct: {}/{} ({:.0f}%)".format(correct, len(dataset), percent))
            # for i in range(b):
            #     plt.scatter(x[i], y[i], s = 20,color = color_dict[idx %9]/255, alpha = alpha_dict[int(idx/9)])
    # plt.axis([xmin, xmax, ymin, ymax])
    plt.show()
    plt.savefig("result1.png")









