import time
import argparse
import pandas as pd
from utils import Logger, Setting, models_load, parse_args, parse_args_boolean
from data_process import context_data_load, context_data_split, context_data_loader
from train import train, test
import wandb

def main(args):

    wandb.login()

    Setting.seed_everything(args.seed)

    ######################## DATA LOAD
    print(f'--------------- {args.model} Load Data ---------------')
    data = context_data_load(args)


    ######################## Train/Valid Split
    print(f'--------------- {args.model} Train/Valid Split ---------------')
    data = context_data_split(data = data)
    data = context_data_loader(args, data)


    ####################### Setting for Log
    setting = Setting()

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)

    logger = Logger(args, log_path)
    logger.save_args()

        
    filename = setting.get_submit_filename(args)

    wandb.init(entity='suggestify_lv2', project="dkt", config=vars(args))
    wandb.run.name = filename[9:-4]
    wandb.run.save()
    

    ######################## Model
    print(f'--------------- INIT {args.model} ---------------')
    model = models_load(args,data)


    ######################## TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    model = train(args, model, data, logger, setting)


    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    predicts = test(args, model, data, setting)


    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    
    if args.model in ('FM', 'FFM'):
        submission['prediction'] = predicts
    else:
        pass

    submission.to_csv(filename, index=False)


if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument


    ############### BASIC OPTION
    arg('--data_path', type=str, default='../../data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, default='FM', choices=['FM', 'FFM'], help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=parse_args_boolean, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=parse_args_boolean, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')

    ############### TRAINING OPTION
    arg('--batch_size', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--epochs', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--lr', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE', choices=['MSE', 'RMSE', 'BCE'], help='손실 함수를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')


    ############### GPU
    arg('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')


    ############### FM, FFM, NCF, WDN, DCN, DeepFM, DeepFFM Common OPTION
    arg('--embed_dim', type=int, default=16, help='FM, FFM, DeepFM, DeepFFM에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--dropout', type=float, default=0.2, help='DeepFM, DeepFFM에서 Dropout rate를 조정할 수 있습니다.')
    arg('--mlp_dims', type=parse_args, default=(16, 16), help='DeepFM, DeepFFM에서 MLP Network의 차원을 조정할 수 있습니다.')


    args = parser.parse_args()
    
    main(args)