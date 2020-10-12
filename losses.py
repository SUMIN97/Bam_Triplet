import torch
import torch.optim as optim
import torchvision
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.cosSimilarity = nn.CosineSimilarity(dim = 1, eps = 1e-6)
        self.margin = margin

        #나중에 삭제
        self.up = 0.0
        self.down = 0.0
        self.result = 0.0
        self.input1 = torch.tensor([])
        self.input2 =  torch.tensor([])

    def forward(self, anchor, positive, negative):
        # anchor = anchor.view(1, len(anchor))
        # positive = anchor.view(1, len(positive))
        # negative = anchor.view(1, len(negative))
        #
        # distance_positive =  1 - self.cosSimilarity(anchor, positive) #가장 비슷:0, 기장다름 :2
        # distance_negative = 1 - self.cosSimilarity(anchor, negative)
        # loss = max(0, self.margin + distance_positive - distance_negative)

        a_p_cosiensimilarity = self.cosineSimilarity(anchor, positive) #target:1
        a_n_cosiensimilarity = self.cosineSimilarity(anchor, negative) #target:-1



        distance_positive = 1 - a_p_cosiensimilarity #target:0 #최악: 2
        distance_negative = 1 - a_n_cosiensimilarity #target:2 #최악:0

        if torch.isnan(distance_positive) or torch.isnan((distance_negative)):
            print("here nan")
            print("input1", self.input1, "input2", self.input2)


        loss = max(0, self.margin + distance_positive - distance_negative) #최악이 2.6 #최선 0
        # print("loss", loss)
        # l2_norm = loss + self.lamda * self.l2_penalty

        return loss

    def cosineSimilarity(self, input1, input2):
        dot = torch.dot(input1, input2)
        a = torch.pow(input1, 2).sum().sqrt()
        b = torch.pow(input2, 2).sum().sqrt()

        self.up = dot
        self.down = a*b
        self.result = self.up/self.down
        self.input1 = input1
        self.input2 = input2

        return  self.result

    def l2_penalty(self):
        return self.parameters.pow().sum()/2























