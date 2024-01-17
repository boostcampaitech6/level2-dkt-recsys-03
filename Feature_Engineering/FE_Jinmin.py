import pandas as pd
from sklearn.preprocessing import LabelEncoder

def feature_engineering(df):
    #라벨 인코딩
    le = LabelEncoder()
    le.fit(df['testId'])
    df['encoded_testId'] = le.transform(df['testId'])


    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

    # testId와 KnowledgeTag, 문제의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    correct_k.columns = ["tag_mean", 'tag_sum']
    correct_i = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum'])
    correct_i.columns = ['item_mean', 'item_sum']

    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, correct_i, on=['assessmentItemID'], how='left')

    # 미래 정보
    df['correct_shift_-3'] = df.groupby('userID')['answerCode'].shift(-3)
    df['correct_shift_-2'] = df.groupby('userID')['answerCode'].shift(-2)
    df['correct_shift_-1'] = df.groupby('userID')['answerCode'].shift(-1)

    # 과거 정답 정보
    df['correct_shift_1'] = df.groupby('userID')['answerCode'].shift(1)
    df['correct_shift_2'] = df.groupby('userID')['answerCode'].shift(2)
    df['correct_shift_3'] = df.groupby('userID')['answerCode'].shift(3)

    #문제 풀이에 사용한 총 시간
    diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    diff = diff.fillna(pd.Timedelta(seconds=0))
    diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
    df['elapsed'] = diff

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
    
    #사용자의 야행성 여부
    df['is_night'] = df['hour_mode'] > 12

    # 시간 정규화
    df['normalized_elapsed'] = df.groupby('userID')['elapsed'].transform(lambda x: (x - x.mean())/x.std())

    # 상대적 시간
    df['relative_time'] = df.groupby('userID').apply(lambda x: x['elapsed'] - x['elapsed'].median()).values

    # 문제 풀이 시간 구간으로 나누기 - 3, 5
    df['time_cut_3'] = pd.cut(df['elapsed'], bins=3)
    df['time_qcut_3'] = pd.qcut(df['elapsed'], q=3)
    df['time_cut_5'] = pd.cut(df['elapsed'], bins=5)
    df['time_qcut_5'] = pd.qcut(df['elapsed'], q=5)

    # 과거에 맞춘 문제 수
    df['shift'] = df.groupby('userID')['answerCode'].shift().fillna(0)
    df['past_correct'] = df.groupby('userID')['shift'].cumsum()
    df.drop(['shift'], axis=1, inplace=True)
    
    # 과거에 해당 문제/시험지/태그를 푼 횟수
    df['past_content_count'] = df.groupby(['userID', 'assessmentItemID']).cumcount()
    df['past_test_count'] = df.groupby(['userID', 'testId']).cumcount()
    df['past_tag_count'] = df.groupby(['userID', 'KnowledgeTag']).cumcount()

    # 과거에 해당 문제/시험지/태그를 맞춘 횟수
    df['shift'] = df.groupby(['userID', 'assessmentItemID'])['answerCode'].shift().fillna(0)
    df['past_content_correct'] = df.groupby(['userID', 'assessmentItemID'])['shift'].cumsum()

    df['shift'] = df.groupby(['userID', 'KnowledgeTag'])['answerCode'].shift().fillna(0)
    df['past_tag_correct'] = df.groupby(['userID', 'KnowledgeTag'])['shift'].cumsum()

    df['shift'] = df.groupby(['userID', 'testId'])['answerCode'].shift().fillna(0)
    df['past_test_correct'] = df.groupby(['userID', 'testId'])['shift'].cumsum()

    # 과거에 해당 문제/시험지/태그 평균 정답률
    df['average_content_correct'] = (df['past_content_correct'] / df['past_content_count']).fillna(0)
    df['average_test_correct'] = (df['past_test_correct'] / df['past_test_count']).fillna(0)
    df['average_tag_correct'] = (df['past_tag_correct'] / df['past_tag_count']).fillna(0)

    # 최근 3개 문제 평균 풀이 시간
    df['mean_time'] = df.groupby(['userID'])['elapsed'].rolling(3).mean().values

    # 문제 번호 추출
    df['itemId'] = df['assessmentItemID'].apply(lambda x: x[7:])


    return df
