import os

import numpy as np
import torch
import wandb

from dkt import trainer
from dkt.args import parse_args
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, set_seeds, logging_conf

from sklearn.model_selection import KFold


logger = get_logger(logging_conf)


def main(args):
    wandb.login()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    
    if args.split_method == "general":
        print("Using general Method")
        train_data, valid_data = preprocess.split_data(data=train_data)
        wandb.init(project=args.wandb_project_name, config=vars(args))
        
        logger.info("Building Model ...")
        model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
        
        logger.info("Start Training ...")
        trainer.run(args=args, train_data=train_data, valid_data=valid_data, model=model)
    
    elif args.split_method == "kfold":
        print("Using k_fold")
        n_splits = args.n_splits
        kfold_auc_list = []
        kf = KFold(n_splits=n_splits)
        
        #---------------------KFOLD 학습 진행---------------------
        for k_th, (train_idx, valid_idx) in enumerate(kf.split(train_data)):
            train_set = torch.utils.data.Subset(train_data, indices = train_idx) # KFold에서 나온 인덱스로 훈련 셋 생성
            val_set = torch.utils.data.Subset(train_data, indices = valid_idx) # KFold에서 나온 인덱스로 검증 셋 생성

            wandb.init(project=args.wandb_project_name, config=vars(args))
            
            logger.info("Building Model ...")
            model: torch.nn.Module = trainer.get_model(args=args).to(args.device)

            trainer.run(args, train_set, val_set, kfold_auc_list, model = model)

            
        #---------------------KFold 결과 출력----------------------
        for i in range(n_splits):
            print(f"Best AUC for {i+1}th fold is : {kfold_auc_list[i]}")
        print(f"The Average AUC of the model is : {sum(kfold_auc_list) / n_splits:.4f}")



if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
