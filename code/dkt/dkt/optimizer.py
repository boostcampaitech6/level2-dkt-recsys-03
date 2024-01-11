import torch
from torch.optim import Adam, AdamW


def get_optimizer(model: torch.nn.Module, args):
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        
    elif args.optimizer == "adamW":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()
    return optimizer
