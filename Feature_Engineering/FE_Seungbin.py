import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

#"DataFrame.dtypes for data must be int, float or bool"

def feature_eng(df: pd.DataFrame):
    df['QID'] = df['assessmentItemID'].str.strip().str[-3:].astype('int')
    df['testNum'] = df['assessmentItemID'].str.strip().str[1:7].astype('int')
    df['testCat'] = df['testId'].str.strip().str[2].astype('int')
    #shift #nan값 처리-> 일단 0으로
    df['shift_1'] = df.groupby('userID')['answerCode'].shift(1).fillna(0)
    df['shift_2'] = df.groupby('userID')['answerCode'].shift(2).fillna(0)
    #문제 푼 시간(초단위)
    df['next_time'] = df.groupby('userID')['Timestamp'].shift(-1)
    df['time'] = (df['next_time'] - df['Timestamp']).dt.seconds
    df = df.drop('next_time',axis=1)
    #time 이상치 처리//위쪽만 처리
    iqr = df['time'].quantile(0.75) - df['time'].quantile(0.25)
    threshold = df['time'].quantile(0.75) + iqr*1.5
    def outlier(v):
        if v > threshold:
            return threshold+10 #이상치 기준 +10
        return v
    df['time'] = df['time'].apply(outlier)

    #time nan 처리 test 필요 #quantile 0.75로 할수도? 마지막 문제니까?
    df['time'] = df['time'].fillna(df.groupby('userID')['time'].transform('median'))

    #푼 시간 누적합
    df['total_time'] = df.groupby('userID')['time'].cumsum()
    #과거에 푼 문제 누적합 #0~t-1
    df['past_Q'] = df.groupby('userID').cumcount()
    #과거에 맞춘 문제 누적합
    df['total_correct'] = df.groupby('userID')['answerCode'].cumsum()
    df['total_correct'] = df.groupby('userID')['total_correct'].shift(1).fillna(0)
    #과거 평균 정답률
    df['avg_correct'] = (df['total_correct']/df['past_Q']).fillna(0)
    #최근 3개 문제 평균 풀이 시간
    df['avg_time_3'] = df.groupby('userID')['time'].rolling(window = 3, min_periods=1).mean().values
    df['avg_time_3'] = df['avg_time_3'].shift(1).fillna(0)
    #최근 3개 문제 평균 정답률 #nan값 처리 -> 일단 0
    df['avg_correct_3'] = df.groupby('userID')['answerCode'].rolling(window = 3, min_periods=1).mean().shift().values
    df['avg_correct_3'] = df['avg_correct_3'].shift(1).fillna(0)
    #add statistics feature
    stat = df[['userID','answerCode']]
    stat = stat.groupby('userID').agg(['mean','std'])
    stat = stat.to_dict()
    for k,v in stat.items():
        feature_name = '_'.join(k)
        df[feature_name] = df['userID'].map(v)
    #시간은 평균대신 중앙값
    df['time_mid'] = df.groupby('userID')['time'].transform('median')
    #시간 정규화
    df['time_norm'] = df.groupby('userID')['time'].transform(lambda x: (x - x.mean())/x.std())
    #유저 평균과 시간 비교
    df['time_rel'] = df.groupby('userID').apply(lambda x: x['time'] - x['time_mid']).values
    #hour
    df['hour'] = df['Timestamp'].transform(lambda x: pd.to_datetime(x, unit='s').dt.hour)
    #정답률 계산에서 data leak을 피하기위해 마지막 시퀀스 제거
    df_del = df.sort_values(by=['userID','Timestamp'], ascending=True).groupby('userID').apply(lambda x: x.iloc[:-1])
    #correct per hour
    hour_dict = df_del.groupby(['hour'])['answerCode'].mean().to_dict()
    df['correct_per_hour'] = df['hour'].map(hour_dict)
    #유저의 테스트/태그/대분류별 정답률 #data leak 고치는 중
    #acc_test_dict = df_del.groupby(['userID','testId'])['answerCode'].transform('mean').to_dict()
    #df['acc_test'] = df['testId'].map(acc_test_dict)
    #df['acc_test'] = df.groupby(['userID','testId'])['answerCode'].transform('mean')
    #df['acc_tag'] = df.groupby(['userID','KnowledgeTag'])['answerCode'].transform('mean')
    #df['acc_cat'] = df.groupby(['userID','testCat'])['answerCode'].transform('mean')
    #전체 문제/테스트/태그/대분류별 정답률
    acc_test_tot_dict = df_del.groupby(['testId'])['answerCode'].mean().to_dict()
    df['acc_test_tot'] = df['testId'].map(acc_test_tot_dict)
    acc_tag_tot_dict = df_del.groupby(['KnowledgeTag'])['answerCode'].mean().to_dict()
    df['acc_tag_tot'] = df['KnowledgeTag'].map(acc_tag_tot_dict)
    acc_cat_tot_dict = df_del.groupby(['testCat'])['answerCode'].mean().to_dict()
    df['acc_cat_tot'] = df['testCat'].map(acc_cat_tot_dict)
    acc_q_tot_dict = df_del.groupby(['assessmentItemID'])['answerCode'].mean().to_dict()
    df['acc_q_tot'] = df['assessmentItemID'].map(acc_q_tot_dict)

    return df