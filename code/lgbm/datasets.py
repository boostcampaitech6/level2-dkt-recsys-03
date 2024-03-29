import os
import random
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

def percentile(x):
    return sum(x)/len(x)

def feature_engineering(df, is_train):
    if os.path.exists("FE_train.csv") and is_train==True:
        df = pd.read_csv("FE_train.csv")
        return df
    elif os.path.exists("FE_test.csv") and is_train==False:
        df = pd.read_csv("FE_test.csv")
        return df

    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])


    ## 기본 정보 ##
    
    # assessmentItemID 분리해서 시험지ID, 시험지 대분류, 시험지 소분류, 문제 번호 생성
    df['testID1'] = df['assessmentItemID'].apply(lambda x: x[1:4])
    df['testID2'] = df['assessmentItemID'].apply(lambda x: x[4:7])
    df['testNum'] = df['assessmentItemID'].apply(lambda x: x[7:])
    
    # 라벨 인코딩
    df['encoded_assessmentItemID'] = LabelEncoder().fit(df['assessmentItemID']).transform(df['assessmentItemID'])
    df['encoded_testId'] = LabelEncoder().fit(df['testId']).transform(df['testId'])
    df['encoded_testID1'] = LabelEncoder().fit(df['testID1']).transform(df['testID1'])
    df['encoded_testID2'] = LabelEncoder().fit(df['testID2']).transform(df['testID2'])
    df['encoded_testNum'] = LabelEncoder().fit(df['testNum']).transform(df['testNum'])    


    ## 시간 정보 ##

    # 문제 풀이 시간 (userID와 testId에 따라 분리, 마지막은 이전 시간으로 결측치 대체)
    df['elapsed'] = (df.groupby(['userID','testId'])['Timestamp'].shift(-1) - df['Timestamp']).apply(lambda x: x.seconds)
    df['elapsed'] = df['elapsed'].ffill().astype('int')
    
    # time 이상치 처리//위쪽만 처리
    iqr = df['elapsed'].quantile(0.75) - df['elapsed'].quantile(0.25)
    threshold = df['elapsed'].quantile(0.75) + iqr*1.5
    def outlier(v):
        if v > threshold:
            return threshold+10 #이상치 기준 +10
        return v
    df['elapsed'] = df['elapsed'].apply(outlier)
    
    # 문제 풀이 시간 중간값
    agg_df = df.groupby('userID')['elapsed'].agg(['median'])
    agg_dict = agg_df.to_dict()
    df['elapsed_median'] = df['userID'].map(agg_dict['median'])

    # 문제를 푸는 시간대
    df['hour'] = df['Timestamp'].transform(lambda x: pd.to_datetime(x, unit='s').dt.hour)

    #시간대별 정답률
    hour_dict = df.groupby(['hour'])['answerCode'].mean().to_dict()
    df['correct_per_hour'] = df['hour'].map(hour_dict)
    
    # 사용자의 주 활동 시간
    mode_dict = df.groupby(['userID'])['hour'].agg(lambda x: pd.Series.mode(x)[0]).to_dict()
    df['hour_mode'] = df['userID'].map(mode_dict)
    
    # 사용자의 야행성 여부
    df['is_night'] = df['hour_mode'] > 12

    # 시간 정규화
    df['normalized_elapsed'] = df['elapsed'].transform(lambda x: (x - x.mean())/x.std())
    df['normalized_elapsed_user'] = df.groupby('userID')['elapsed'].transform(lambda x: (x - x.mean())/x.std())
    df['normalized_elapsed_test'] = df.groupby('testId')['elapsed'].transform(lambda x: (x - x.mean())/x.std())
    df['normalized_elapsed_user_test'] = df.groupby(['userID','testId'])['elapsed'].transform(lambda x: (x - x.mean()) / x.std())

    # 상대적 시간
    df['relative_time'] = df.groupby('userID').apply(lambda x: x['elapsed'] - x['elapsed'].median()).values

    # 문제 풀이 시간 구간으로 나누기 - 3, 5
    df['time_cut_3'] = pd.cut(df['elapsed'], bins=3)
    df['time_qcut_3'] = pd.qcut(df['elapsed'], q=3)
    df['time_cut_5'] = pd.cut(df['elapsed'], bins=5)
    df['time_qcut_5'] = pd.qcut(df['elapsed'], q=5)
    
    # 시간 구간 라벨 인코딩
    df['encoded_time_cut_3'] = LabelEncoder().fit(df['time_cut_3']).transform(df['time_cut_3'])
    df['encoded_time_qcut_3'] = LabelEncoder().fit(df['time_qcut_3']).transform(df['time_qcut_3'])
    df['encoded_time_cut_5'] = LabelEncoder().fit(df['time_cut_5']).transform(df['time_cut_5'])
    df['encoded_time_qcut_5'] = LabelEncoder().fit(df['time_qcut_5']).transform(df['time_qcut_5'])

    # 월, 요일, 주차
    df['month'] = df['Timestamp'].dt.month
    df['wday'] = df['Timestamp'].dt.day_of_week
    df['week_num'] = df['Timestamp'].dt.isocalendar().week
    month_mean = df.groupby('month')['answerCode'].mean().to_dict()
    df['month_mean'] = df['month'].map(month_mean)


    ## 누적 정보 ##

    # 유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

    # User 별 문제 수 / 정답 수 / 정답률
    df['past_user_count'] = df.groupby('userID').cumcount()
    df['shift'] = df.groupby('userID')['answerCode'].shift().fillna(0)
    df['past_user_correct'] = df.groupby('userID')['shift'].cumsum()
    df['average_user_correct'] = (df[f'past_user_correct'] / df[f'past_user_count']).fillna(0)
    
    # Feature 별 문제 수 / 정답 수 / 정답률
    feature_list = ['assessmentItemID','testId','testID1','testID2','testNum','KnowledgeTag']
    for feature in feature_list:
        # Feature 별 계산
        df[f'past_{feature}_count'] = df.groupby(feature).cumcount()
        df['shift'] = df.groupby(feature)['answerCode'].shift().fillna(0)
        df[f'past_{feature}_correct'] = df.groupby(feature)['shift'].cumsum()
        df[f'average_{feature}_correct'] = (df[f'past_{feature}_correct'] / df[f'past_{feature}_count']).fillna(0)
    
    # User와 Feature 별 문제 수 / 정답 수 / 정답률
    feature_list = ['assessmentItemID','testId','testID1','testID2','testNum','KnowledgeTag']
    for feature in feature_list:
        df[f'past_user_{feature}_count'] = df.groupby(['userID', feature]).cumcount()
        df['shift'] = df.groupby(['userID', feature])['answerCode'].shift().fillna(0)
        df[f'past_user_{feature}_correct'] = df.groupby(['userID', feature])['shift'].cumsum()
        df[f'average_user_{feature}_correct'] = (df[f'past_user_{feature}_correct'] / df[f'past_user_{feature}_count']).fillna(0)
    
    df = df.drop('shift', axis=1)


    ## 전체 정보 ##

    # testId와 KnowledgeTag, 문제의 전체 정답률은 한번에 계산 (아래 데이터는 제출용 데이터셋에 대해서도 재사용)
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum', 'count'])
    correct_t.columns = ['test_mean', 'test_sum', 'test_count']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum', 'count'])
    correct_k.columns = ['tag_mean', 'tag_sum', 'tag_count']
    correct_i = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum', 'count'])
    correct_i.columns = ['item_mean', 'item_sum', 'item_count']

    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, correct_i, on=['assessmentItemID'], how='left')

    df['test_normalized_mean'] = (df['test_mean'] - df['test_mean'].mean()) / df['test_mean'].std()
    df['tag_normalized_mean'] = (df['tag_mean'] - df['tag_mean'].mean()) / df['tag_mean'].std()
    df['item_normalized_mean'] = (df['item_mean'] - df['item_mean'].mean()) / df['item_mean'].std()

    # 문제 번호별 평균 정답률
    average_testNum = df.groupby('testNum')['answerCode'].mean().to_dict()
    df['testNum_mean'] = df['testNum'].map(average_testNum)


    # 문제 대분류와 유저의 대분류별 정답률, 풓이 횟수, 풀이시간 평균
    df['assessment_class'] = df['assessmentItemID'].map(lambda x:int(x[2]))
    df['assessment_class_sum'] = df.groupby(['userID', 'assessment_class'])['answerCode'].cumsum()
    df['assessment_class_count'] = df.groupby(['userID', 'assessment_class'])['answerCode'].cumcount()
    df['assessment_class_mean'] = df['assessment_class_sum'] / df['assessment_class_count']
    mean_solved_time = df.groupby('assessment_class')['normalized_elapsed_user'].mean().to_dict()
    df['assessment_class_mean_time'] = df['assessment_class'].map(mean_solved_time)

    ## Shift 정보 ##

    # 미래 정보
    df['correct_shift_-3'] = df.groupby('userID')['answerCode'].shift(-3)
    df['correct_shift_-2'] = df.groupby('userID')['answerCode'].shift(-2)
    df['correct_shift_-1'] = df.groupby('userID')['answerCode'].shift(-1)

    # 과거 정답 정보
    df['correct_shift_1'] = df.groupby('userID')['answerCode'].shift(1)
    df['correct_shift_2'] = df.groupby('userID')['answerCode'].shift(2)
    df['correct_shift_3'] = df.groupby('userID')['answerCode'].shift(3)
    df['correct_shift_4'] = df.groupby('userID')['answerCode'].shift(4)
    df['correct_shift_5'] = df.groupby('userID')['answerCode'].shift(5)

    # 미래 정답 정보 (userID와 testId에 따라 분리)
    df['correct_ut_shift_-1'] = df.groupby(['userID','testId'])['answerCode'].shift(-1)
    df['correct_ut_shift_-2'] = df.groupby(['userID','testId'])['answerCode'].shift(-2)
    df['correct_ut_shift_-3'] = df.groupby(['userID','testId'])['answerCode'].shift(-3)
    
    # 과거 정답 정보 (userID와 testId 기준) 
    df['correct_ut_shift_1'] = df.groupby(['userID','testId'])['answerCode'].shift(1)
    df['correct_ut_shift_2'] = df.groupby(['userID','testId'])['answerCode'].shift(2)
    df['correct_ut_shift_3'] = df.groupby(['userID','testId'])['answerCode'].shift(3)
    df['correct_ut_shift_4'] = df.groupby(['userID','testId'])['answerCode'].shift(4)
    df['correct_ut_shift_5'] = df.groupby(['userID','testId'])['answerCode'].shift(5)

    #문제 정답률 shift
    df['average_assessmentItemID_correct_shift_1'] = df.groupby(['userID','testId'])['average_assessmentItemID_correct'].shift(1)
    df['average_assessmentItemID_correct_shift_2'] = df.groupby(['userID','testId'])['average_assessmentItemID_correct'].shift(2)
    df['average_assessmentItemID_correct_shift_3'] = df.groupby(['userID','testId'])['average_assessmentItemID_correct'].shift(3)
    df['average_assessmentItemID_correct_shift_1'] = df['average_assessmentItemID_correct_shift_1'].ffill()
    df['average_assessmentItemID_correct_shift_2'] = df['average_assessmentItemID_correct_shift_2'].ffill()
    df['average_assessmentItemID_correct_shift_3'] = df['average_assessmentItemID_correct_shift_3'].ffill()

    # 시간 Shift
    shift_columns = ['normalized_elapsed', 'normalized_elapsed_user', 'normalized_elapsed_test', 'normalized_elapsed_user_test']
    shift_steps = [1, 2, 3]
    for col in shift_columns:
        for step in shift_steps:
            shifted_col_name = f'{col}_shift_{step}'  # Feature 생성
            df[shifted_col_name] = df.groupby(['userID', 'testId'])[col].shift(step)  # 시프트 적용
            df[shifted_col_name] = df[shifted_col_name].ffill()  # 결측값 채우기

    # 최근 3문제를 푼 시간의 평균
    df['avg_elapsed_3'] = df.groupby('userID')['elapsed'].rolling(window = 3, min_periods=1).mean().values
    df['avg_elapsed_3'] = df['avg_elapsed_3'].shift(1).fillna(0)

    # 최근 3개 문제 풀이 갯수, 정답 횟수, 정답률
    df['recent_sum'] = df.groupby('userID')['elapsed'].rolling(window = 3, min_periods=1).sum().values
    df['recent_sum'] = df['recent_sum'].shift(1).fillna(0)
    df['recent_mean'] = df.groupby('userID')['elapsed'].rolling(window = 3, min_periods=1).mean().values
    df['recent_mean'] = df['recent_mean'].shift(1).fillna(0)    



    ## Relative Feature 생성 ##
    AnswerRate = df.groupby('assessmentItemID')['answerCode'].mean()
    df['testIDAnswerRate'] = df['assessmentItemID'].map(AnswerRate)
    df['relative_answered_correctly'] = df['answerCode']-df['testIDAnswerRate']
    
    df = df.drop(['assessmentItemID','testId','testID1','testID2','testNum','Timestamp','time_cut_3','time_qcut_3','time_cut_5','time_qcut_5'], axis=1)
    
    if is_train==True:
        df.to_csv('FE_train.csv')

        df = pd.read_csv('FE_train.csv')
    elif is_train==False:
        df.to_csv('FE_test.csv')
        df = pd.read_csv('FE_test.csv')
    
    return df

def custom_train_test_split(df, ratio=0.8, split=True):

    users = list(set(df['userID']))
    random.shuffle(users)
    user_ids = users[:round(ratio*len(users))]

    test = df[df['userID'].isin(user_ids) & (df['userID'] != df['userID'].shift(-1))]
    train = df.drop(test.index)
    return train, test


def prepare_dataset(args):
    train_df = pd.read_csv(os.path.join(args.data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_data.csv"))

    test_df['answerCode'].replace(-1, 0.5, inplace=True) #예측 타깃 answercode 0.5로 변경
    total_df = pd.concat([train_df, test_df], ignore_index=True) # train,test를 합하기
    total_df.sort_values(by=['userID','Timestamp'], inplace=True)
   
    test_df = feature_engineering(total_df, False) #train,test로 test의 FE 진행
    if args.using_train is True: #학습을 할때 Test의 앞부분 (예측 타깃 제외) 활용 여부
        train_df = test_df[test_df['answerCode'] != 0.5]
    train_df, valid_df = custom_train_test_split(train_df)
    
    test_df = test_df[test_df['answerCode'] == 0.5] #예측 타깃(0.5)만 남기지
    
    X_train = train_df[args.feats]
    y_train = train_df['answerCode']
    
    X_valid = valid_df[args.feats]
    y_valid = valid_df['answerCode']
    
    X_test = test_df[args.feats]
    
    train_dataset = lgb.Dataset(X_train, y_train)
    valid_dataset = lgb.Dataset(X_valid, y_valid)
    
    return train_dataset, valid_dataset, X_test