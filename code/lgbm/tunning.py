# Hyper parameter tunning
# 사용방법 : python tunning.py
import os
import wandb
import lightgbm as lgb

from args import parse_args
from utils import set_seeds, get_logger, logging_conf
from datasets import prepare_dataset
from trainer import train, inference
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier

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

def main(args): 
    train_dataset, valid_dataset, X_test = prepare_dataset(args)

    X_train = train_dataset.data
    y_train = train_dataset.label
    
    # LGBM 모델 초기화
    model = LGBMClassifier( 
        objective='binary',
        #num_leaves=args.num_leaves,
        #min_data_in_leaf=args.min_data_in_leaf,
        #max_depth=args.max_depth,
        #max_bin=args.max_bin,
        #learning_rate=args.lr,
        #num_iterations=args.n_iterations,
        metric=['auc'],
        seed=args.seed
    )   

    # 탐색할 하이퍼파라미터 공간 정의
    param_grid = {
        'num_iterations': [100,300,500],
        'learning_rate': [0.25, 0.05, 0.1],        
        'num_leaves': [32, 64, 128],
        'min_data_in_leaf': [1, 5, 10],
        'max_depth': [16, 32, 64],
        'max_bin': [50, 100, 150]
    }

    # Randomized 탐색 초기화 
    random_search = RandomizedSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=2, n_iter=200, random_state=args.seed)
    # 학습 데이터를 사용하여 랜덤 탐색 실행
    random_search.fit(X_train, y_train)
    # 최적의 하이퍼파라미터 출력
    print('Best parameters: ', random_search.best_params_)
    # 최고 성능 출력
    print('Best score: ', random_search.best_score_)
    
    '''
    # 그리드 탐색 초기화
    grid_search = GridSearchCV(model, param_grid, cv=2, scoring='accuracy', verbose=2)      
    # 학습 데이터를 사용하여 그리드 탐색 실행
    grid_search.fit(X_train, y_train)
    # 최적의 하이퍼파라미터 출력
    print('Best parameters: ', grid_search.best_params_)
    # 최고 성능 출력
    print('Best score: ', grid_search.best_score_)
    #Fitting 2 folds for each of 729 candidates, totalling 1458 fits mean=100s
    '''

if __name__=='__main__':
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args)