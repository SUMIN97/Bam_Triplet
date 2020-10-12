import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from .trainer import fit
import numpy as np
cuda = torch.cuda.is_available()



batch_size = 1
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from .net import  TripleNet, StyleNet, ContentNet
from .losses import TripletLoss

margin = 1.
model = TripleNet(StyleNet(), ContentNet())
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100