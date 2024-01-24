import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(color_codes=True)
import pandas as pd


DATA_PATH = 'data/train_data.csv'

dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}

df = pd.read_csv(DATA_PATH, dtype=dtype, parse_dates=['Timestamp'])
df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

# 각 'assessmentItemID'에 대한 정답 수를 계산합니다.
correct_counts = df[df['answerCode'] == 1]['assessmentItemID'].value_counts()
# 각 'assessmentItemID'에 대한 전체 문제 수를 계산합니다.
total_counts = df['assessmentItemID'].value_counts()
# 정답률을 계산합니다.
correct_rates = correct_counts / total_counts
# 결과를 출력합니다.
print(correct_rates[correct_rates>0.7])


# 문제 풀이 시간 (userID와 testId에 따라 분리, 마지막은 이전 시간으로 결측치 대체)
df['elapsed'] = (df.groupby(['userID','testId'])['Timestamp'].shift(-1) - df['Timestamp']).apply(lambda x: x.seconds)
df['elapsed'] = df['elapsed'].ffill().astype('int')

# 산점도 그리기
plt.figure(figsize=(8, 6))
sns.scatterplot(x='elapsed', y='answerCode', data=df)
plt.title('Elapsed Time vs Answer Code')
plt.xlabel('Elapsed Time (seconds)')
plt.ylabel('Answer Code')
plt.savefig('소요시간-정답률.png')

# 'elapsed' 값의 범위를 10개의 동일한 길이의 구간으로 나눕니다.
df['elapsed_bins'] = pd.cut(df['elapsed'], bins=100)

# 각 구간에 속하는 데이터의 수를 계산합니다.
bin_counts = df['elapsed_bins'].value_counts().sort_index()

# 결과를 출력합니다.
print(bin_counts)

# 각 'elapsed_bins' 구간에 대한 'answerCode'의 평균(= 정답률)을 계산합니다.
answer_rate = df.groupby('elapsed_bins')['answerCode'].mean()

# 결과를 출력합니다.
print(answer_rate)
### Feature Engineering ###


## 시간 정규화 & 날짜 생성 (소요시간 : 5min) ##
df['noramalized_time'] = df.groupby(['userID','testId'])['Timestamp'].transform(lambda x: (x - x.mean()) / x.std())
df['date'] = df['Timestamp'].dt.date

## Relative Feature 생성 ##
AnswerRate = df.groupby('assessmentItemID')['answerCode'].mean()
df['testIDAnswerRate'] = df['assessmentItemID'].map(AnswerRate)
df['relative_answered_correctly'] = df['answerCode']-df['testIDAnswerRate']

## 직전 정답 생성 ##
df['prevUserID'] = df['userID'].shift(1)
df['prevAnswerCode'] = df['answerCode'].shift(1)
df.loc[df['userID'] != df['prevUserID'], 'prevAnswerCode'] = pd.NA
df.drop('prevUserID', axis=1, inplace=True)

#df.to_csv('data_FE.csv', index=False)



### EDA ###


## 시간에 따른 정답률 ##
# normalized_time을 0.5 단위의 구간으로 나눕니다.
df['normalized_time_bin'] = pd.cut(df['noramalized_time'], bins=np.arange(df['noramalized_time'].min(), df['noramalized_time'].max() + 0.5, 0.5))
grouped_by_time_bin = df.groupby('normalized_time_bin')['answerCode'].mean()
print(grouped_by_time_bin)
# 계산된 평균 정답률을 막대 그래프로 시각화합니다.
grouped_by_time_bin.plot(kind='bar', figsize=(10, 6))
# 그래프 제목과 축 레이블을 설정합니다.
plt.title('Average Answer Code by Normalized Time Bin')
plt.xlabel('Normalized Time Bin')
plt.ylabel('Average Answer Code')
# 그래프를 화면에 표시합니다.
plt.savefig('소요시간-정답률.png')


## 같은 문제를 다시 풀었을 경우 ##
df_filtered = df[df.duplicated(subset=['userID', 'assessmentItemID'], keep=False)]
grouped_data = df_filtered.groupby(['userID', 'assessmentItemID'])
print(f" 동일한 유저,item는 총 {len(grouped_data)}개 있으며, 각 item은 {grouped_data.size().mean()}번씩 반복되었다")
# 동일한 유저,item는 총 45119개 있으며, 각 item은 2.018484452226335번씩 반복되었다
second_answers = df_filtered.groupby(['userID', 'assessmentItemID']).nth(1)['answerCode']
# 각 그룹에서 첫 번째 'answerCode'를 얻기 위한 변환을 수행합니다.
first_answer = df.groupby(['userID', 'assessmentItemID'])['answerCode'].transform('first')
# 첫 번째 'answerCode'가 1인 행들만 필터링합니다.
df_first_answer_1 = df[(first_answer == 1)]
df_first_answer_0 = df[(first_answer == 0)]
# 이제 이 데이터를 다시 그룹화하여 각 그룹의 두 번째 'answerCode' 값을 추출합니다.
# 'nth(1)'는 각 그룹의 두 번째 행을 의미합니다.
second_answers_1 = df_first_answer_1.groupby(['userID', 'assessmentItemID']).nth(1)['answerCode']
second_answers_0 = df_first_answer_0.groupby(['userID', 'assessmentItemID']).nth(1)['answerCode']
# userID와 assessmentItemID로 그룹화한 다음 각 그룹의 사이즈를 계산합니다.
#first_answer_1.groupby(['userID', 'assessmentItemID']).size().value_counts()

# 결과를 출력합니다.
print(f"동일한 유저가 동일한 문제를 풀었을 경우\n  1.갯수:{len(second_answers)}  2.정답률:{second_answers.mean()}")
print(f"처음 문제를 맞추고 동일한 문제를 풀었을 경우\n  1.갯수:{len(second_answers_1)}  2.정답률:{second_answers_1.mean()}")
print(f"처음 문제를 틀리고 동일한 문제를 풀었을 경우\n  1.갯수:{len(second_answers_0)}  2.정답률:{second_answers_0.mean()}")
# 동일한 유저가 동일한 문제를 풀었을 경우      1.갯수:45119  2.정답률:0.6451384117555797
# 처음 문제를 맞추고 동일한 문제를 풀었을 경우  1.갯수:30493  2.정답률:0.6967172793755944
# 처음 문제를 틀리고 동일한 문제를 풀었을 경우  1.갯수:14626  2.정답률:0.5376042663749487



# 문제를 다시풀었을때와 시간의 관계
second_answers_1 = df_first_answer_0.groupby(['userID', 'assessmentItemID']).nth(1)[['answerCode','time_diff']]
# 'time_diff'를 일(day) 단위로 변환합니다.
second_answers_1['time_diff_days'] = second_answers_1['time_diff'].dt.days
# 최소 일수와 최대 일수를 구합니다.
min_days = second_answers_1['time_diff_days'].min()
max_days = second_answers_1['time_diff_days'].max()
# 일 단위로 bin을 만듭니다. 예를 들어, 1일 간격으로 나누기
bins = np.arange(min_days, max_days + 60, 60)
labels = [f'Day {int(day)}' for day in bins[:-1]]
# 'time_diff_days'를 이용해 binning합니다.
second_answers_1['day_bin'] = pd.cut(second_answers_1['time_diff_days'], bins=bins, labels=labels, include_lowest=True)
# 각 일별 bin에 대한 정답률을 계산합니다.
bin_correctness = second_answers_1.groupby('day_bin')['answerCode'].mean()
# 정답률을 막대 그래프로 시각화합니다.
plt.figure(figsize=(14, 7))
bin_correctness.plot(kind='bar')
plt.title('The elapsed time - accuracy upon reattempting')
plt.xlabel('Days')
plt.ylabel('Answer rate')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('문제를 맞춘후 지난 시간 - 다시풀었을때 정답률.png')



# 'prevUserID'라는 새로운 컬럼을 생성하여 각 행에 대해 이전 행의 'userID' 값을 저장합니다.
df['prevUserID'] = df['userID'].shift(1)
# 'prevAnswerCode'라는 새로운 컬럼을 생성하여 각 행에 대해 이전 행의 'answerCode' 값을 저장합니다.
# 'shift' 함수는 데이터를 한 행씩 아래로 이동시킵니다.
df['prevAnswerCode'] = df['answerCode'].shift(1)
# 'userID'가 변경될 때마다 'prevAnswerCode'에 NaN을 할당합니다.
df.loc[df['userID'] != df['prevUserID'], 'prevAnswerCode'] = pd.NA
# 이전 'answerCode'가 1인 행들만 필터링합니다.
filtered_1 = df[df['prevAnswerCode'] == 1]
filtered_0 = df[df['prevAnswerCode'] == 0]

print(f"이전 문제를 맞춘 경우 같은 사용자의 다음 문제도 맞출 확률: {filtered_1['answerCode'].mean():.2f}")
#이전 문제를 맞춘 경우 같은 사용자의 다음 문제도 맞출 확률: 0.77
print(f"이전 문제를 틀린 경우 같은 사용자의 다음 문제도 맞출 확률: {filtered_0['answerCode'].mean():.2f}")
#이전 문제를 틀린 경우 같은 사용자의 다음 문제도 맞출 확률: 0.44


## Sequential answerCode ##
#직전의 answerCode와 현재의 answercode가 동일할 확률이 높디
diff_count = (df['answerCode'].shift() != df['answerCode']).sum()
same_count = (df['answerCode'].shift() == df['answerCode']).sum()
print("이전 값과 다른 경우 count:", diff_count)
#이전 값과 다른 경우 count: 695955
print("동일한 경우 count:", same_count)
#동일한 경우 count: 1570631
diff_count2 = ((df['answerCode'].shift() == df['answerCode'].shift(2)) & (df['answerCode'].shift() != df['answerCode'])).sum()
same_count2 = ((df['answerCode'].shift() == df['answerCode'].shift(2)) & (df['answerCode'].shift() == df['answerCode'])).sum()
print("직전값과 2번째 이전값이 같고, 직전값과 현재값이 다른 경우 count:", diff_count2)
#직전값과 2번째 이전값이 같고, 직전값과 현재값이 다른 경우 count: 375400
print("직전값과 2번째 이전값이 같고, 직전값과 현재값이 같은 경우 count:", same_count2)
#직전값과 2번째 이전값이 같고, 직전값과 현재값이 같은 경우 count: 1195230


# answerCode가 1, 1, 0, 0인 경우를 카운트
change_count = (df['answerCode'].shift() == 1) & (df['answerCode'] == 1) & (df['answerCode'].shift(-1) == 0) & (df['answerCode'].shift(-2) == 0)
change_count = change_count.sum()
print("answerCode가 1, 1, 0, 0인 경우의 개수:", change_count)



## 문제 순서에 따른 정답률 ##
df['questionSequence'] = df['assessmentItemID'].str[-3:]
grouped_df = df.groupby('questionSequence').agg({'Sequence': 'count', 'Answer Rate': 'mean'})
grouped_df.columns = ['Sequence', 'Answer Rate']
print(grouped_df)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.bar(grouped_df.index, grouped_df['Answer Rate'])
plt.xlabel('Question Sequence')
plt.ylabel('Answer Rate')
plt.title('Answer Rate per Question Sequence')
plt.xticks(rotation=45)  # x축 라벨 회전
plt.tight_layout()
plt.savefig('문제순서-정답률.png')



## 문항을 풀수록 실력이 늘어나는가? ##
# 누적합
_cumsum = df.loc[:, ['userID', 'relative_answered_correctly']].groupby('userID').agg({'relative_answered_correctly': 'cumsum'})
# 누적갯수
_cumcount = df.loc[:, ['userID', 'relative_answered_correctly']].groupby('userID').agg({'relative_answered_correctly': 'cumcount'}) + 1
cum_ans = _cumsum #/ _cumcount
cum_ans['userID'] = df['userID']
window_sizes = [30, 70, 100, 120]

#푼 문항의 갯수가 중앙값 부근인 10명의 학샐들
fig, ax = plt.subplots()
ax.set_title('Median 10 Students Answer Rate among their Study')
ax.set_xlabel('# of Questions solved')
ax.set_ylabel('Relative Answered Correctly')

samples = df.groupby('userID').agg({'assessmentItemID': 'count'}).sort_values(by='assessmentItemID').index[7442//2-5:7442//2+5]
for idx in samples:
    cum_ans[cum_ans['userID'] == idx]['relative_answered_correctly'].reset_index(drop=True).plot(ax=ax, label=f'User {idx}')
ax.legend(bbox_to_anchor=(1, 1))
plt.savefig('푼문제수-상대정답률(푼문제수 중앙값 10명).png')



## 사용자 분석 ##
def percentile(s):
    return np.sum(s) / len(s)
stu_groupby = df.groupby('userID').agg({
    'assessmentItemID': 'count',
    'relative_answered_correctly': 'sum'
})
stu_groupby.describe()
## 같은 문항수를 푼 학생들을 묶은 그래프
itemnum_ans = stu_groupby.groupby('assessmentItemID').mean()
itemnum_ans['num_items'] = itemnum_ans.index
fig, ax = plt.subplots()
sns.regplot(data=itemnum_ans, x='num_items', y='relative_answered_correctly',
           line_kws={"color": "orange"}, scatter_kws={'alpha':0.6}, ax=ax)

ax.set_title('# of Questions - Answer Rate')
ax.set_xlabel('# of Questions')
ax.set_ylabel('Relative Answered Correctly')
plt.savefig('푼문할수-상대정답률(같은 문항수 유저 묶기).png')

# "# of Questions"와 "Relative Answered Correctly" 간의 상관계수 계산
correlation = itemnum_ans['num_items'].corr(itemnum_ans['relative_answered_correctly'])
print(f"# of Questions와 Relative Answered Correctly 간의 상관계수: {correlation:.3f}")

# 태그를 푼 사용자의 수와 정답률 사이의 상관계수 출력
tag_groupby = df.groupby('태그').agg({'userID': 'count', 'relative_answered_correctly': 'mean'})
correlation = tag_groupby.corr()['relative_answered_correctly']['userID']

print(f"태그를 푼 사용자의 수와 정답률 사이의 상관계수: {correlation:.3f}")


## Bin 추가
itemnum_ans = stu_groupby.groupby('assessmentItemID').mean()
bins = 300 #Bin : 나누는 구간 수
itemnum_ans['bins'] = pd.cut(
    itemnum_ans.index,
    [i * (itemnum_ans.index.max() - itemnum_ans.index.min()) // bins for i in range(bins)]
)
itemnum_ans = itemnum_ans.groupby('bins').mean()
itemnum_ans['mid'] = list(map(lambda x: (x.left + x.right)//2, itemnum_ans.index))
fig, ax = plt.subplots()
sns.regplot(data=itemnum_ans, x='mid', y='relative_answered_correctly',
           line_kws={"color": "orange"}, scatter_kws={'alpha': 0.6}, ax=ax)
ax.set_title(f'# of Items - Answer Rate | bins={bins}')
ax.set_xlabel('# of Items')
ax.set_ylabel('Relative Answered Correctly')
plt.savefig('푼문항수-상대정답률(bin).png')






#첫번째 단어 확인 -> A만 있음
print("assessmentItemID의 첫번째 단어 :",df['assessmentItemID'].apply(lambda x: x[0]).unique())

#문항 일련번호 내 시험지 번호 일치 확인 -> 문항 일련번호 1~6 = 시험지 번호 1~3+마지막3
print("assessmentItemID(1~6)와 TestID(1~3+-3~-1) 일치 여부 :",
    (df['assessmentItemID'].apply(lambda x: x[1:7]) == df['testId'].apply(lambda x: x[1:4]+x[7:])).value_counts())

#시험지 번호 가운데 -> 000
print("TestID 가운데 값 :",df['testId'].apply(lambda x: x[4:7]).unique())

## 사용자 분석 ##
def percentile(s):
    return np.sum(s) / len(s)

stu_groupby = df.groupby('userID').agg({
    'assessmentItemID': 'count',
    'answerCode': percentile
})
stu_groupby.describe()

## 사용자 분석 ##
#사용자 문항 개수 도수분포표
fig, ax = plt.subplots()
stu_groupby['assessmentItemID'].hist(bins=20, ax=ax)
ax.set_title('Student # of items Histogram')
ax.set_xlabel('# of items solved')
ax.set_ylabel('Count')
ax.axvline(stu_groupby['assessmentItemID'].mean(), color='red') #Red line : Mean
ax.grid(visible=True)
plt.savefig('사용자 문항 개수 도수분포표.png')
#사용자 정답률 도수분포표
fig, ax = plt.subplots()
stu_groupby['answerCode'].hist(bins=20)
ax.set_title('Student Answer Rate Histogram')
ax.set_xlabel('Answer Rate')
ax.set_ylabel('Count')
ax.axvline(stu_groupby['answerCode'].mean(), color='red') 
ax.grid(visible=True)
plt.savefig('사용자 정답률 도수분포표.png')

## 문항별 분석 ##
prob_groupby = df.groupby('assessmentItemID').agg({
    'userID': 'count',
    'answerCode': percentile
})
prob_groupby.describe()

# 문항별 정답률 도수분포표
fig, ax = plt.subplots()
prob_groupby['answerCode'].hist(bins=20)
ax.set_title('Item Answer Rate Histogram')
ax.set_xlabel('Answer Rate')
ax.set_ylabel('Count')
ax.axvline(prob_groupby['answerCode'].mean(), color='red')
ax.grid(visible=True)
plt.savefig('문항별 정답률 도수분포표.png')

## 시험지별 분석 ##
test_groupby = df.groupby('testId').agg({
    'userID': 'count',
    'answerCode': percentile
})
test_groupby.describe()
#시험지 별 정답률 분석
fig, ax = plt.subplots()
test_groupby['answerCode'].hist(bins=20)
ax.set_title('Test Answer Rate Histogram')
ax.set_xlabel('Answer Rate')
ax.set_ylabel('Count')
ax.axvline(test_groupby['answerCode'].mean(), color='red')
ax.grid(visible=True)
plt.savefig('시험지 별 정답률 분석.png')

### 일반적인 EDA
## 푼 문항수와 정답률의 관계 ##
#분포도
g = sns.lmplot(
    data=stu_groupby,
    x='assessmentItemID',
    y='answerCode',
    scatter_kws={'alpha':0.3},
    line_kws={"color": "orange"}
)
g.set_xlabels('# of Questions solved')
g.set_ylabels('Answer Rate')
g.set(xlim=(-30, 1900))

ax = plt.gca()
ax.set_title('# of Questions - Answer Rate')
plt.savefig('정답률과 푼문제수 상관관계.png')
#인과관계
print(f"정답률과 문제를 푼 개수 사이 인과관계: {stu_groupby.corr()['assessmentItemID']['answerCode']:.3f}") #0.168
#비교
fig, ax = plt.subplots()

ax.set_title('Comparison between Answer Rates of students\nwho solved more than average and the other')
stu_num_mean = stu_groupby['assessmentItemID'].mean()
stu_groupby[stu_groupby['assessmentItemID'] >= stu_num_mean]['answerCode'].hist(
    bins=20, ax=ax, alpha=0.7, stacked=True, density=1, label='Solved more than Average'
)
stu_groupby[stu_groupby['assessmentItemID'] < stu_num_mean]['answerCode'].hist(
    bins=20, ax=ax, alpha=0.7, stacked=True, density=1, label='Solved less than Average'
)
ax.legend()
plt.savefig('정답률과 푼문제수 비교.png')

## 더 많이 노출된 태그가 정답률이 높은가?
tag_groupby = df.groupby('KnowledgeTag').agg({
    'userID': 'count', #해당 문제태그를 푼 사용자의 수 count
    'answerCode': percentile
})
tag_groupby.describe()

#분포도
g = sns.lmplot(
    data=tag_groupby,
    x='userID',
    y='answerCode',
    scatter_kws={'alpha':0.5},
    line_kws={"color": "orange"}
)
g.set_xlabels('# of Tags Being solved')
g.set_ylabels('Answer Rate')
g.set(xlim=(-30, 14500))

ax = plt.gca()
ax.set_title('# of Tags Exposed - Answer Rate')
plt.savefig('정답률과 문제노출빈도 상관관계.png')

#상관관계
tag_groupby
print(f"태그를 풀었던 사용자의 수와 정답률 사이 상관관계: {tag_groupby.corr()['answerCode']['userID']:.3f}") #0.376

#비교
fig, ax = plt.subplots()

tag_ans_mean = tag_groupby['userID'].mean()
ax.set_xlabel('Answer Rate')
ax.set_ylabel('Scaled Counts')
ax.set_title('Comparison between Answer rates of Tags\nbeing solved more than average and the other')

tag_groupby[tag_groupby['userID'] >= tag_ans_mean]['answerCode'].hist(
    ax=ax, alpha=0.7, bins=15, density=1, stacked=True, label='Solved more than Average'
)
tag_groupby[tag_groupby['userID'] < tag_ans_mean]['answerCode'].hist(
    ax=ax, alpha=0.55, bins=15, density=1, stacked=True, label='Solved less than Average'
)
ax.legend()
plt.savefig('정답률과 문제노출빈도 비교.png')

## 문항을 풀수록 실력이 늘어나는가? ##
# 누적합
_cumsum = df.loc[:, ['userID', 'answerCode']].groupby('userID').agg({'answerCode': 'cumsum'})
# 누적갯수
_cumcount = df.loc[:, ['userID', 'answerCode']].groupby('userID').agg({'answerCode': 'cumcount'}) + 1
cum_ans = _cumsum / _cumcount
cum_ans['userID'] = df['userID']
window_sizes = [30, 70, 100, 120]

#푼 문항의 갯수가 중앙값 부근인 10명의 학샐들
fig, ax = plt.subplots()
ax.set_title('Median 10 Students Answer Rate among their Study')
ax.set_xlabel('# of Questions solved')
ax.set_ylabel('Answer Rate')

samples = df.groupby('userID').agg({'assessmentItemID': 'count'}).sort_values(by='assessmentItemID').index[7442//2-5:7442//2+5]
for idx in samples:
    cum_ans[cum_ans['userID'] == idx]['answerCode'].reset_index(drop=True).plot(ax=ax, label=f'User {idx}')
ax.legend(bbox_to_anchor=(1, 1))
plt.savefig('푼문제수-정답률(푼문제수 중앙값 10명).png')

#정답률 중앙값 부근인 10명의 학생들
fig, ax = plt.subplots()
ax.set_title('Median 10 Answer Rates Student among their Study')
ax.set_xlabel('# of Questions solved')
ax.set_ylabel('Answer Rate')

samples = df.groupby('userID').agg({'answerCode': percentile}).sort_values(by='answerCode').index[7442//2-5:7442//2+5]
for idx in samples:
    cum_ans[cum_ans['userID'] == idx]['answerCode'].reset_index(drop=True).plot(ax=ax, label=f'User {idx}')
ax.legend(bbox_to_anchor=(1, 1))
plt.savefig('푼문제수-정답률(정답률 중앙값 10명).png')

#Window size 적용
def plot_rolled_answerrate(userID, ax, window_sizes=[70, 100, 120]):
    
    ax.set_title(f'Students Answer Rate among their Study - User {userID}')
    ax.set_xlabel('# of Questions solved')
    ax.set_ylabel('Answer Rate')

    cum_ans[cum_ans['userID'] == userID]['answerCode'].reset_index(drop=True).plot(
        ax=ax, label=f'Without Window', linewidth=3)

    for wdw_sz in window_sizes:    
        (df[df.userID == userID]['answerCode'].rolling(wdw_sz).sum().reset_index(drop=True) / wdw_sz).plot(
            ax=ax, label=f'Window size {wdw_sz}', alpha=0.8)

    ax.legend()

fig, ax = plt.subplots()
plot_rolled_answerrate(500, ax)
plt.savefig('푼문제수-정답률(한사용자).png')

##문항을 푸는 데 걸린 시간과 정답률 사이의 관계
diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
diff = diff.fillna(pd.Timedelta(seconds=0))
diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

df['elapsed'] = diff

elapsed_answer = df.groupby('elapsed').agg({'answerCode': percentile, 'userID': 'count'})
elapsed_hist = elapsed_answer[elapsed_answer['userID'] > 100]

fig, ax = plt.subplots()
g = sns.regplot(x=elapsed_hist.index[:-1], y=elapsed_hist.answerCode.values[:-1],
            scatter_kws={'alpha':0.5}, line_kws={"color": "orange"}, ax=ax)
ax.set_title('Elapsed Time vs. Answer Rate')
ax.set_xlabel('Elapsed Time (< 650s)')
ax.set_ylabel('Answer Rate')
ax.axvline(22, color='r')
ax.set_xlim(-10, 650)
plt.savefig('시간-정답률.png')

## 같은 문항수를 푼 학생들을 묶은 그래프
itemnum_ans = stu_groupby.groupby('assessmentItemID').mean()
itemnum_ans['num_items'] = itemnum_ans.index
fig, ax = plt.subplots()
sns.regplot(data=itemnum_ans, x='num_items', y='answerCode',
           line_kws={"color": "orange"}, scatter_kws={'alpha':0.6}, ax=ax)

ax.set_title('# of Questions - Answer Rate')
ax.set_xlabel('# of Questions')
ax.set_ylabel('Answer Rate')
plt.savefig('푼문할수-정답률(같은 문항수 유저 묶기).png')

## Bin 추가
itemnum_ans = stu_groupby.groupby('assessmentItemID').mean()
bins = 300 #Bin : 나누는 구간 수
itemnum_ans['bins'] = pd.cut(
    itemnum_ans.index,
    [i * (itemnum_ans.index.max() - itemnum_ans.index.min()) // bins for i in range(bins)]
)
itemnum_ans = itemnum_ans.groupby('bins').mean()
itemnum_ans['mid'] = list(map(lambda x: (x.left + x.right)//2, itemnum_ans.index))
fig, ax = plt.subplots()
sns.regplot(data=itemnum_ans, x='mid', y='answerCode',
           line_kws={"color": "orange"}, scatter_kws={'alpha': 0.6}, ax=ax)
ax.set_title(f'# of Items - Answer Rate | bins={bins}')
ax.set_xlabel('# of Items')
ax.set_ylabel('Answer Rate')
plt.savefig('푼문항수-정답률(bin).png')
