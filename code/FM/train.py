import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.optim import SGD, Adam
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Tuple


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss


def train(args, model, dataloader, logger, setting):
    minimum_loss = 999999999
    if args.loss_fn == 'MSE':
        loss_fn = MSELoss()
    elif args.loss_fn == 'RMSE':
        loss_fn = RMSELoss()
    elif args.loss_fn == 'BCE':
        loss_fn = BCEWithLogitsLoss()
    else:
        pass

    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'ADAM':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        pass

    for epoch in tqdm.tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        batch = 0
        total_preds = []
        total_targets = []

        for idx, data in enumerate(dataloader["train_dataloader"]):
            x, y = data[0].to(args.device), data[1].to(args.device)
            y_hat = model(x)
            loss = loss_fn(y.float(), y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            y_hat = torch.sigmoid(y_hat)
            total_preds.append(y_hat.detach())
            total_targets.append(y.detach())
            batch += 1

        total_preds = torch.concat(total_preds).cpu().numpy()
        total_targets = torch.concat(total_targets).cpu().numpy()
        auc, acc = get_metric(targets=total_targets, preds=total_preds)
        valid_loss, valid_auc, valid_acc = valid(args, model, dataloader, loss_fn)

        print(f"Train AUC: {auc} ACC: {acc}")
        print(f"VALID AUC: {valid_auc} ACC: {valid_acc}")
        print(f"Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}")

        wandb.log(dict(epoch=epoch,
                       train_loss_epoch=total_loss/batch,
                       train_auc_epoch=auc,
                       train_acc_epoch=acc,
                       valid_loss_epoch=valid_loss/batch,
                       valid_auc_epoch=valid_auc,
                       valid_acc_epoch=valid_acc))

        if minimum_loss > valid_loss:
            minimum_loss = valid_loss
            os.makedirs(args.saved_model_path, exist_ok=True)
            model_file_path = f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pt'
            torch.save(model.state_dict(), model_file_path)
    return model

def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0
    total_preds = []
    total_targets = []

    for idx, data in enumerate(dataloader["valid_dataloader"]):
        x, y = data[0].to(args.device), data[1].to(args.device)
        y_hat = model(x)
        loss = loss_fn(y.float(), y_hat)
        total_loss += loss.item()
        y_hat = torch.sigmoid(y_hat)
        total_preds.append(y_hat.detach())
        total_targets.append(y.detach())
        batch += 1
    valid_loss = total_loss / batch
    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    
    return valid_loss, auc, acc


def test(args, model, dataloader, setting):
    predicts = list()
    if args.use_best_model == True:
        model.load_state_dict(torch.load(f'./saved_models/{setting.save_time}_{args.model}_model.pt'))
    else:
        pass
    model.eval()

    for idx, data in enumerate(dataloader['test_dataloader']):
        x = data[0].to(args.device)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    return predicts


def get_metric(targets: np.ndarray, preds: np.ndarray) -> Tuple[float]:
    auc = roc_auc_score(y_true=targets, y_score=preds)
    acc = accuracy_score(y_true=targets, y_pred=np.where(preds >= 0.5, 1, 0))
    return auc, acc
