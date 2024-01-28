import numpy as np
import random
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import StandardScaler


def acc_map(x: float) -> int:
    if x < 0.20:
        return 0
    elif x >= 0.20 and x < 0.40:
        return 1
    elif x >= 0.40 and x < 0.60:
        return 2
    elif x >= 0.60 and x < 0.80:
        return 3
    else:
        return 4

def process_context_data(train_df, test_df):

    # userID,assessmentItemID,testId,answerCode,Timestamp,KnowledgeTag
    train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
    test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
    train_df['hour'] = train_df['Timestamp'].transform(lambda x: pd.to_datetime(x, unit='s').dt.hour)
    test_df['hour'] = test_df['Timestamp'].transform(lambda x: pd.to_datetime(x, unit='s').dt.hour)

    context_df = pd.concat([train_df, test_df])
    
    correct_t = train_df.groupby(['userID'])['answerCode'].agg(['mean'])
    correct_t.columns = ["user_acc"]
    train_df = pd.merge(train_df, correct_t, on=['userID'], how="left")
    

    train_df['user_acc'] = train_df['user_acc'].apply(acc_map)
    user_df = train_df[train_df['userID'] != train_df['userID'].shift(-1)]
    test_df = test_df.merge(user_df[['userID', 'user_acc']], on='userID', how='left')
    

    # 인덱싱 처리
    tag2idx = {v:k for k,v in enumerate(context_df['KnowledgeTag'].unique())}
    test2idx = {v:k for k,v in enumerate(context_df['testId'].unique())}
    hour2idx = {v:k for k,v in enumerate(context_df['hour'].unique())}

    train_df['KnowledgeTag'] = train_df['KnowledgeTag'].map(tag2idx)
    train_df['testId'] = train_df['testId'].map(test2idx)
    train_df['hour'] = train_df['hour'].map(hour2idx)
    
    test_df['KnowledgeTag'] = test_df['KnowledgeTag'].map(tag2idx)
    test_df['testId'] = test_df['testId'].map(test2idx)
    test_df['hour'] = test_df['hour'].map(hour2idx)
    

    
    train_df.drop(['Timestamp'], axis=1, inplace=True)
    test_df.drop(['Timestamp'], axis=1, inplace=True)

    

    idx = {
        "tag2idx":tag2idx,
        "test2idx":test2idx,
        'hour2idx':hour2idx
    }

    return idx, train_df, test_df


def context_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    data_dir = '../../data'
    train_path = os.path.join(data_dir, 'train_data.csv')
    test_path = os.path.join(data_dir, 'test_data.csv')
    sub_path = os.path.join(data_dir,'sample_submission.csv')

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sub_path)

    train = pd.concat([train, test[test['answerCode']!=-1]])
    
    test.drop(test[test['answerCode'] != -1].index, inplace=True)

    train = train.drop_duplicates(subset=["userID", "assessmentItemID"], keep="last")
    train = train.reset_index(drop=True)

    ids = train['userID'].unique()
    items = train['assessmentItemID'].unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2item = {idx:id for idx, id in enumerate(items)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    item2idx = {id:idx for idx, id in idx2item.items()}

    train['userID'] = train['userID'].map(user2idx)
    test['userID'] = test['userID'].map(user2idx)

    train['assessmentItemID'] = train['assessmentItemID'].map(item2idx)
    test['assessmentItemID'] = test['assessmentItemID'].map(item2idx)


    idx, context_train, context_test = process_context_data(train, test)
    field_dims = np.array([len(user2idx), len(item2idx),len(idx['tag2idx']), len(idx['test2idx']),len(idx['hour2idx']), 5], dtype=np.uint32)

    data = {
            'train':context_train,
            'test':context_test.drop(['answerCode'], axis=1),
            'field_dims':field_dims,
            'sub':sub,
            'idx2user':idx2user,
            'idx2item':idx2item,
            'user2idx':user2idx,
            }

    return data



def context_data_split(data, ratio=0.8, split=True):

    df = data['train']

    random.seed(42)
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.shuffle(users)

    max_train_data_len = ratio*len(df)
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)


    train = df[df['userID'].isin(user_ids)]
    valid = df[df['userID'].isin(user_ids) == False]

    #test데이터셋은 각 유저의 마지막 interaction만 추출
    train = pd.concat([train, valid[valid['userID'] == valid['userID'].shift(-1)]])
    valid = valid[valid['userID'] != valid['userID'].shift(-1)]

    X_train = train.drop(columns=["answerCode"])
    X_valid = valid.drop(columns=["answerCode"])
    y_train = train["answerCode"]
    y_valid = valid["answerCode"]

    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data


def context_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
