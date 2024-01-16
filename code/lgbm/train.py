import os

import wandb
from args import parse_args
from utils import set_seeds, get_logger, logging_conf
from datasets import prepare_dataset
from trainer import train, inference

logger = get_logger(logging_conf)

def main(args): 
    wandb.login()
    wandb.init(project='dkt-lgbm', config=vars(args))
    wandb.run.name = args.run
    
    set_seeds(args.seed)
    
    logger.info("Preparing data ...")
    train_dataset, valid_dataset, X_test = prepare_dataset(args)
    
    logger.info("Running Model ...")
    model = train(args, train_dataset, valid_dataset)
    
    logger.info("Saving Prediction ...")
    inference(args, model, X_test)
    
if __name__=='__main__':
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args)