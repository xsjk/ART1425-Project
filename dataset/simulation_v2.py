import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


class MLPModel(pl.LightningModule):
    def __init__(self, 
                input_dim, 
                output_dim=1,
                hidden_dim=[64, 64, 64],
                dropout=0.4,
                gamma=0.001
                ):
        super().__init__()

        self.gamma = gamma

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
        loss = nn.functional.mse_loss(y_pred, y)
        
        for param in self.parameters():
            loss += self.gamma * torch.norm(param, 2)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True),
                "monitor": "val_loss",
                "frequency": 100,
            },
        }
    
from .preprocess_v2 import X_sec_back_t_train, X_indoor_train

sec_back_t_model = MLPModel(input_dim=X_sec_back_t_train.shape[1])
indoor_model = MLPModel(input_dim=X_indoor_train.shape[1])


