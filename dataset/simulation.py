import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import pandas as pd

default_dropout = 0
default_gamma = 1e-6
default_lr = 1e-2

# sec_back_t
# V0: [64, 64, 64, 64, 64], dropout=0, gamma=1e-4, lr=1e-2 -> 0.0091
# V1: [64, 32, 16, 8, 4], dropout=0, gamma=1e-4, lr=1e-2 -> 0.0170
# V2: [128, 64, 32, 16, 8], dropout=0, gamma=1e-4, lr=1e-2 -> 0.0143
# V3: [32, 32, 32, 32, 32], dropout=0, gamma=1e-4, lr=1e-2 -> 0.0141
# V4: [32, 16, 8, 4, 2], dropout=0, gamma=1e-4, lr=1e-2 -> 0.0124
# V0: [32, 16, 8, 4, 2], dropout=0, gamma=1e-4, lr=1e-2, lr_scheduler=StepLR(step_size=10000, gamma=0.9) -> 0.0184
# V1: [64, 64, 64, 64, 64], dropout=0, gamma=1e-4, lr=1e-2, lr_scheduler=StepLR(step_size=10000, gamma=0.9) -> 0.0126
# V2: [64, 32, 16, 8, 4], dropout=0, gamma=1e-4, lr=1e-2, lr_scheduler=StepLR(step_size=10000, gamma=0.9) -> 0.0112
# V3: [64, 32, 16, 8, 4], dropout=0, gamma=1e-4, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65) -> 0.0157
# V4: [64, 64, 64, 64, 64], dropout=0, gamma=1e-4, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65) -> 0.0229
# V5: [32, 16, 8, 4, 2], dropout=0, gamma=1e-4, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65)

# V6: [64, 32, 16, 8, 4], dropout=0, gamma=1e-6, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65) -> 0.0098 (best)
# V7: [64, 32, 16, 8, 4], dropout=0.2, gamma=1e-6, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65) -> 0.0708
# V8: [64, 32, 16, 8, 4], dropout=0, gamma=1e-5, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65) -> 0.0103
# V9: [64, 32, 16, 8, 4], dropout=0, gamma=1e-7, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65) -> 0.0204
# V10: [64, 32, 16, 8, 4], dropout=0.1, gamma=1e-6, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65) -> 0.0378
# V11: [64, 32, 16, 8, 4], dropout=0, gamma=1e-8, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65) -> 0.0130

# indoor_t
# V0: [64, 32, 16, 8, 4], dropout=0, gamma=1e-6, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65 -> 0.0184 (best)
# V4: [32, 16, 8, 4, 2], dropout=0, gamma=1e-6, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65) -> 0.0182


# branch

# joint
# V7: [64, 32, 16, 8, 4], dropout=0, gamma=1e-6, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65) -> (0.0077, 0.0206)
# V8: [512, 256, 128, 64, 32], dropout=0, gamma=1e-6, lr=1e-2, lr_scheduler=StepLR(step_size=5000, gamma=0.65) -> (0.0175, 0.0559)

class MLPModel(pl.LightningModule):
    def __init__(self, 
                input_dim, 
                output_dim=1,
                hidden_dim=[512, 256, 128, 64, 32, 16, 8, 4],
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.65)
        return [optimizer], [scheduler]
    
    def predict(self, state: pd.Series):
        return self(torch.tensor(state.values, dtype=torch.float32, device=self.device)).detach().cpu().numpy()


class BranchModel(pl.LightningModule):
    def __init__(self, 
                input_dim, 
                output_dim=(1, 1),
                hidden_dim=([1024, 512, 256, 128, 64], [32, 16, 8, 4], [32, 16, 8, 4]),
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.65)
        return [optimizer], [scheduler]
