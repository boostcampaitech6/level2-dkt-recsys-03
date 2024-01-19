import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from wandb.lightgbm import wandb_callback, log_summary

def get_params(args):
    config = {
            'seed':args.seed,
            'objective': 'binary',
            #'num_leaves': args.num_leaves,
            #'min_data_in_leaf':args.min_data_in_leaf,
            #'max_depth':args.max_depth,
            #'early_stopping_round':args.early_stopping_round,
            #'max_bin':args.max_bin,
            #'learning_rate':args.lr,
            #'num_iterations':args.n_iterations,
            'metric':['auc']
            }
    return config

def train(args, lgb_train, lgb_valid):
    X_valid = lgb_valid.data
    y_valid = lgb_valid.label
    
    model = lgb.train(
    get_params(args),
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    num_boost_round=500, 
    callbacks = [wandb_callback()]
    )
    
    log_summary(model, save_model_checkpoint=True)
    
    preds = model.predict(X_valid)
    acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_valid, preds)
    print(f'VALID AUC : {auc} ACC : {acc}\n')
    
    return model

def inference(args, model, X_test):
    preds = model.predict(X_test)
    write_path = os.path.join(args.output_dir, "submission.csv")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(preds):
            w.write('{},{}\n'.format(id,p))
            
    return