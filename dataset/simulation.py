import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

default_dropout = 0.2
default_gamma = 1e-4
default_lr = 1e-2

class MLPModel(pl.LightningModule):
    def __init__(self, 
                input_dim, 
                output_dim=1,
                hidden_dim=[64, 32, 16, 8, 4],
                dropout=default_dropout,
                gamma=default_gamma,
                lr=default_lr):
        super().__init__()

        self.gamma = gamma
        self.lr = lr

        layers = []
        for in_dim, out_dim in zip([input_dim] + hidden_dim[:-1], hidden_dim):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.linear = nn.Sequential(*layers)
        self.save_hyperparameters()

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred.squeeze(), y)
        
        for param in self.parameters():
            loss += self.gamma * torch.norm(param, 2)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred.squeeze(), y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class BranchModel(pl.LightningModule):
    def __init__(self, 
                input_dim, 
                output_dim=(1, 1),
                hidden_dim=([64, 32, 16], [8, 4], [8, 4]),
                dropout=default_dropout,
                gamma=default_gamma,
                lr=default_lr):
        super().__init__()

        self.gamma = gamma
        self.lr = lr

        layers = []
        for in_dim, out_dim in zip([input_dim] + hidden_dim[0][:-1], hidden_dim[0]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.Dropout(dropout))

        self.branch1 = nn.Sequential(*layers)

        layers = []
        for in_dim, out_dim in zip([hidden_dim[0][-1]] + hidden_dim[1][:-1], hidden_dim[1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim[1][-1], output_dim[0]))
        self.branch2 = nn.Sequential(*layers)

        layers = []
        for in_dim, out_dim in zip([hidden_dim[0][-1] + 1] + hidden_dim[2][:-1], hidden_dim[2]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim[2][-1], output_dim[1]))
        self.branch3 = nn.Sequential(*layers)

        self.save_hyperparameters()

    def forward(self, x):
        latent = self.branch1(x)
        y1 = self.branch2(latent)
        y2 = self.branch3(torch.cat([latent, x[:, -1].unsqueeze(1)], dim=1))
        return y1, y2

    def training_step(self, batch, batch_idx):
        x, y = batch
        y1, y2 = y.split(1, dim=1)

        y1_pred, y2_pred = self(x)
        loss1 = F.mse_loss(y1_pred.squeeze(), y1)
        loss2 = F.mse_loss(y2_pred.squeeze(), y2)
        self.log('train_loss1', loss1)
        self.log('train_loss2', loss2)

        loss = loss1 + loss2
        for param in self.parameters():
            loss += self.gamma * torch.norm(param, 2)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y1, y2 = y.split(1, dim=1)

        y1_pred, y2_pred = self(x)
        loss1 = F.mse_loss(y1_pred.squeeze(), y1)
        loss2 = F.mse_loss(y2_pred.squeeze(), y2)
        self.log('val_loss1', loss1)
        self.log('val_loss2', loss2)

        loss = loss1 + loss2
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
