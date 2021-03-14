import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models.resnet import ResNet18
from models.mobilenetv2 import MobileNetV2
import argparse



class Model(pl.LightningModule):
    def __init__(self, pe=0):
        super().__init__()
        self.pe = pe
        self.log('pe', pe)
        # self.net = ResNet18(pe)
        self.net = MobileNetV2(pe)
        self.lr = 0.1
        self.wd = 5e-4

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.net(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x_hat = self.net(x)
        loss = F.cross_entropy(x_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x_hat = self.net(x)
        loss = F.cross_entropy(x_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd, nesterov=True)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, epochs=200, steps_per_epoch=391)
        scheduler = {"scheduler": scheduler, "interval" : "step" }
        return {'optimizer':optimizer, 'lr_scheduler':scheduler}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--pe', default=0, type=int)
    args = parser.parse_args()

    pl.seed_everything(990121)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)

    pe = args.pe
    print(pe)
    model = Model(pe)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='ckpt',
        save_last=True,
        save_top_k=1,
        filename='mobilenetv2-' + ('pe' if pe else 'vanilla') + '-{epoch:03d}-{val_loss:.2f}' 
    )

    trainer = pl.Trainer(gpus=-1, max_epochs=200, callbacks=[checkpoint_callback])
    trainer.fit(model, trainloader, testloader)
