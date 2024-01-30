import os
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_names", default="lgbm,xgboost,catboost,lgcn,ffm,lqtr,gru,bert", type=str, help="make a list of csv filenames")
    args = parser.parse_args()
    return args


def main(args):
    file_names = args.file_names.split(sep=',')
    data_list = []; data_name = []
    
    # outputs 폴더에 모델 별 prediction 파일 추가
    for file_name in file_names:
        data = pd.read_csv(f'./outputs/{file_name}.csv')
        data_list.append(data)
        data_name.append(file_name)
    
    # prediction을 binary로 변경
    for data, name in zip(data_list, data_name):
        data.loc[data['prediction']<0.5,'prediction'] = 0
        data.loc[data['prediction']>=0.5,'prediction'] = 1
        data.to_csv(f'./outputs_voting/{name}_voting.csv')


if __name__ == "__main__":
    args = parse_args()
    os.makedirs('./outputs_voting', exist_ok=True)
    main(args=args)
