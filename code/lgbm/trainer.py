import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from wandb.lightgbm import wandb_callback, log_summary
from wandb import Table
import wandb
import matplotlib.pyplot as plt
import shap

def get_params(args):
    config = {
            'seed':args.seed,
            'objective': 'binary',
            'num_leaves': args.num_leaves,
            'min_data_in_leaf':args.min_data_in_leaf,
            'max_depth':args.max_depth,
            #'early_stopping_round':args.early_stopping_round,
            'max_bin':args.max_bin,
            'learning_rate':args.lr,
            'num_iterations':args.n_iterations,
            'metric':['auc']
            }
    return config

def feature_importance(model, valid):
    
    # split 기반 피처 중요도 계산 및 로깅
    feat_imps_split = model.feature_importance(importance_type='split')
    feats = model.feature_name()
    fi_data_split = [[feat, feat_imp] for feat, feat_imp in zip(feats, feat_imps_split)]
    table_split = wandb.Table(data=fi_data_split, columns=["Feature", "Split Importance"])
    wandb.log({"Feature Importance (Split)": wandb.plot.bar(table_split, "Feature", "Split Importance", title="Feature Importance (Split)")}, commit=False)

    # gain 기반 피처 중요도 계산 및 로깅
    feat_imps_gain = model.feature_importance(importance_type='gain')
    fi_data_gain = [[feat, feat_imp] for feat, feat_imp in zip(feats, feat_imps_gain)]
    table_gain = wandb.Table(data=fi_data_gain, columns=["Feature", "Gain Importance"])
    wandb.log({"Feature Importance (Gain)": wandb.plot.bar(table_gain, "Feature", "Gain Importance", title="Feature Importance (Gain)")}, commit=False)

    # Shapley Value 계산 및 로깅
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(valid)
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values[1], valid, plot_type="dot", feature_names=feats)
    wandb.log({"shap_dot_plot": wandb.Image(plt)}, commit=False)
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values[1], valid, plot_type="bar", feature_names=feats)
    wandb.log({"shap_summary_plot": wandb.Image(plt)}, commit=False)


def train(args, lgb_train, lgb_valid):
    X_valid = lgb_valid.data
    y_valid = lgb_valid.label
    
    model = lgb.train(
    get_params(args),
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    #num_boost_round=500, 
    callbacks = [wandb_callback()]
    )
    
    feature_importance(model, X_valid) #Feature importance를 로깅
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