import os
import random
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

def percentile(x):
    return sum(x)/len(x)

def feature_engineering(df):

    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)

    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    correct_a = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum', 'count'])
    correct_a.columns = ["assessment_mean", 'assessment_sum','assessment_count']
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum', 'count'])
    correct_t.columns = ["test_mean", 'test_sum', 'test_count']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum', 'count'])
    correct_k.columns = ["tag_mean", 'tag_sum', 'tag_count']

    df = pd.merge(df, correct_a, on='assessmentItemID', how='left')
    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    
    #문항 번호와 평균 정답률
    df['qNum'] = df['assessmentItemID'].map(lambda x:int(x[-3:]))
    correct_q = df.groupby('qNum').apply(lambda x:percentile(x['answerCode'])).to_dict()
    df['qNum_mean'] = df['qNum'].map(correct_q)
    
    #문제 풀이 추정 시간 
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['solvedTime'] = df.groupby('userID')['Timestamp'].diff().dt.total_seconds().shift(-1).fillna(0)
    
    #주 활동 시간대와 현 시간대
    df['hour'] = df['Timestamp'].dt.hour
    main_hours = df.groupby('userID').apply(lambda x:x['hour'].mode()[0])
    df['mainHour'] = df['userID'].map(main_hours)
    
    return df

def custom_train_test_split(df, ratio=0.7, split=True):

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
    test = df[df['userID'].isin(user_ids) == False]

    #test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test

def prepare_dataset(args):
    train_df = pd.read_csv(os.path.join(args.data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_data.csv"))
    
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    train_df, valid_df = custom_train_test_split(train_df)
    
    X_train = train_df[args.feats]
    y_train = train_df['answerCode']
    
    X_valid = valid_df[args.feats]
    y_valid = valid_df['answerCode']
    
    X_test = test_df[args.feats]
    
    train_dataset = lgb.Dataset(X_train, y_train)
    valid_dataset = lgb.Dataset(X_valid, y_valid)
    
    return train_dataset, valid_dataset, X_test