import torch
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


import dataset.preprocess as preprocess
import dataset.simulation as simulation

import argparse
import pytorch_lightning as pl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=1250)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    devices = [args.device] if args.device is not None else 'auto'

    trainer = pl.Trainer(max_epochs=args.epochs, devices=devices)

    match args.model:
        case 'split.sec_back_t':
            print('Training split.sec_back_t model')
            model = simulation.MLPModel(input_dim=preprocess.X_sec_back_t_train.shape[1], lr=args.lr)
            trainer.fit(model, 
                        preprocess.sec_back_t_train_dataloader, 
                        preprocess.sec_back_t_val_dataloader)
        case 'split.indoor':
            print('Training split.indoor model')
            model = simulation.MLPModel(input_dim=preprocess.X_indoor_train.shape[1], lr=args.lr)
            trainer.fit(model, 
                        preprocess.indoor_train_dataloader, 
                        preprocess.indoor_val_dataloader)   
        case 'branch':
            print('Training branch model')
            model = simulation.BranchModel(input_dim=preprocess.X_branch_train.shape[1], lr=args.lr)
            trainer.fit(model, 
                        preprocess.branch_train_dataloader, 
                        preprocess.branch_val_dataloader)   
        case 'joint':
            print('Training joint model')
            model = simulation.MLPModel(input_dim=preprocess.X_branch_train.shape[1], output_dim=2, lr=args.lr)
            trainer.fit(model, 
                        preprocess.branch_train_dataloader, 
                        preprocess.branch_val_dataloader)
        case _:
            raise ValueError(f'Unknown model {args.model}')
