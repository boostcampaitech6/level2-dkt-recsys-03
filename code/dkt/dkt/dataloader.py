import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self,
                   data: np.ndarray,
                   ratio: float = 0.7,
                   shuffle: bool = True,
                   seed: int = 0) -> Tuple[np.ndarray]:
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]
        return data_1, data_2

    def __save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s: str):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)
        return df

    def __feature_engineering(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        # TODO: Fill in if needed
        
        if os.path.exists(os.path.join(self.args.data_dir, "FE_train.csv")) and is_train==True:
            df = pd.read_csv(os.path.join(self.args.data_dir, "FE_train.csv"))
            return df
        elif os.path.exists(os.path.join(self.args.data_dir, "FE_test.csv")) and is_train==False:
            df = pd.read_csv(os.path.join(self.args.data_dir, "FE_test.csv"))
            return df

        #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        df.sort_values(by=['userID','Timestamp'], inplace=True)

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # assessmentItemID 분리해서 시험지ID, 시험지 대분류, 시험지 소분류, 문제 번호 생성
        df['testID1'] = df['assessmentItemID'].apply(lambda x: x[1:4])
        df['testID2'] = df['assessmentItemID'].apply(lambda x: x[4:7])
        df['testNum'] = df['assessmentItemID'].apply(lambda x: x[7:])
        
        #라벨 인코딩
        df['encoded_assessmentItemID'] = LabelEncoder().fit(df['assessmentItemID']).transform(df['assessmentItemID'])
        df['encoded_testId'] = LabelEncoder().fit(df['testId']).transform(df['testId'])
        df['encoded_testID1'] = LabelEncoder().fit(df['testID1']).transform(df['testID1'])
        df['encoded_testID2'] = LabelEncoder().fit(df['testID2']).transform(df['testID2'])
        df['encoded_testNum'] = LabelEncoder().fit(df['testNum']).transform(df['testNum'])    

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
        
        # 미래 정답 정보 (userID와 testId에 따라 분리)
        df['correct_ut_shift_-1'] = df.groupby(['userID','testId'])['answerCode'].shift(-1)
        df['correct_ut_shift_-2'] = df.groupby(['userID','testId'])['answerCode'].shift(-2)
        df['correct_ut_shift_-3'] = df.groupby(['userID','testId'])['answerCode'].shift(-3)
        
        # 과거 정답 정보 (userID와 testId 기준)
        df['correct_ut_shift_1'] = df.groupby(['userID','testId'])['answerCode'].shift(1)
        df['correct_ut_shift_2'] = df.groupby(['userID','testId'])['answerCode'].shift(2)
        df['correct_ut_shift_3'] = df.groupby(['userID','testId'])['answerCode'].shift(3)

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
        
        # 시간 구간 라벨 인코딩
        df['encoded_time_cut_3'] = LabelEncoder().fit(df['time_cut_3']).transform(df['time_cut_3'])
        df['encoded_time_qcut_3'] = LabelEncoder().fit(df['time_qcut_3']).transform(df['time_qcut_3'])
        df['encoded_time_cut_5'] = LabelEncoder().fit(df['time_cut_5']).transform(df['time_cut_5'])
        df['encoded_time_qcut_5'] = LabelEncoder().fit(df['time_qcut_5']).transform(df['time_qcut_5'])

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
            df[f'past_user_{feature}_count'] = df.groupby(feature).cumcount()
            df['shift'] = df.groupby(feature)['answerCode'].shift().fillna(0)
            df[f'past_user_{feature}_correct'] = df.groupby(feature)['shift'].cumsum()
            df[f'average_user_{feature}_correct'] = (df[f'past_user_{feature}_correct'] / df[f'past_user_{feature}_count']).fillna(0)
        
        df = df.drop('shift', axis=1)
        
        # 최근 3문제를 푼 시간의 평균
        df['avg_elapsed_3'] = df.groupby('userID')['elapsed'].rolling(window = 3, min_periods=1).mean().values
        df['avg_elapsed_3'] = df['avg_elapsed_3'].shift(1).fillna(0)
        
        ################### 시우 ######################
        # 문제 번호별 평균 정답률
        average_testNum = df.groupby('testNum')['answerCode'].mean().to_dict()
        df['testNum_mean'] = df['testNum'].map(average_testNum)
        
        # 월, 요일, 주차
        df['month'] = df['Timestamp'].dt.month
        df['wday'] = df['Timestamp'].dt.day_of_week
        df['week_num'] = df['Timestamp'].dt.isocalendar().week
        month_mean = df.groupby('month')['answerCode'].mean().to_dict()
        df['month_mean'] = df['month'].map(month_mean)
        
        # 문제 대분류와 유저의 대분류별 정답률, 풓이 횟수, 풀이시간 평균
        df['assessment_class'] = df['assessmentItemID'].map(lambda x:int(x[2]))
        df['assessment_class_sum'] = df.groupby(['userID', 'assessment_class'])['answerCode'].cumsum()
        df['assessment_class_count'] = df.groupby(['userID', 'assessment_class'])['answerCode'].cumcount()
        df['assessment_class_mean'] = df['assessment_class_sum'] / df['assessment_class_count']
        mean_solved_time = df.groupby('assessment_class')['normalized_elapsed'].mean().to_dict()
        df['assessment_class_mean_time'] = df['assessment_class'].map(mean_solved_time)
        
        # 최근 3개 문제 풀이 갯수, 정답 횟수, 정답률
        df['recent_sum'] = df.groupby('userID')['elapsed'].rolling(window = 3, min_periods=1).sum().values
        df['recent_sum'] = df['recent_sum'].shift(1).fillna(0)
        df['recent_mean'] = df.groupby('userID')['elapsed'].rolling(window = 3, min_periods=1).mean().values
        df['recent_mean'] = df['recent_mean'].shift(1).fillna(0)
        
        ################### 재원 ######################
        ## 시간 정규화 & 날짜 생성 (소요시간 : 5min) ##
        df['noramalized_time'] = df.groupby(['userID','testId'])['Timestamp'].transform(lambda x: (x - x.mean()) / x.std())
        ## Relative Feature 생성 ##
        AnswerRate = df.groupby('assessmentItemID')['answerCode'].mean()
        df['testIDAnswerRate'] = df['assessmentItemID'].map(AnswerRate)
        df['relative_answered_correctly'] = df['answerCode']-df['testIDAnswerRate']
        
        df = df.drop(['assessmentItemID','testId','testID1','testID2','testNum','Timestamp','time_cut_3','time_qcut_3','time_cut_5','time_qcut_5'], axis=1)
        
        if is_train==True:
            df.to_csv(os.path.join(self.args.data_dir, 'FE_train.csv'))
            df = pd.read_csv(os.path.join(self.args.data_dir, 'FE_train.csv'))
        elif is_train==False:
            df.to_csv(os.path.join(self.args.data_dir, 'FE_test.csv'))
            df = pd.read_csv(os.path.join(self.args.data_dir, 'FE_test.csv'))
        
        return df

    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_tests = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tags = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                )
            )
        )
        return group.values

    def load_train_data(self, file_name: str) -> None:
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name: str) -> None:
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, args):
        self.data = data
        self.max_seq_len = args.max_seq_len

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]
        
        # Load from data
        test, question, tag, correct = row[0], row[1], row[2], row[3]
        data = {
            "test": torch.tensor(test + 1, dtype=torch.int),
            "question": torch.tensor(question + 1, dtype=torch.int),
            "tag": torch.tensor(tag + 1, dtype=torch.int),
            "correct": torch.tensor(correct, dtype=torch.int),
        }

        # Generate mask: max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len:]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len-seq_len:] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        data["mask"] = mask
        
        # Generate interaction
        interaction = data["correct"] + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1)
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        data["interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}
        return data

    def __len__(self) -> int:
        return len(self.data)


def get_loaders(args, train: np.ndarray, valid: np.ndarray) -> Tuple[torch.utils.data.DataLoader]:
    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader
