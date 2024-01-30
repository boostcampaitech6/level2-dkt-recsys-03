# LightGCN
<br>

## 모델 선정
Knowledge Tracing에서는 문제에 대한 사용자의 정답 여부를 예측한다.  
이는 추천 시스템에서 아이템에 대한 사용자의 선호도를 예측하는 작업과 유사하기 때문에 Knowlege Tracing에 추천 모델을 활용 수 있다.  
LightGCN은 NGCF에서 feature transformation과 nonlinear activation을 제거하여 학습의 효율성과 성능을 높인 CF 모델이다.  
그래프 구조를 통해 사용자와 문제 간의 정답 여부에 대한 관계를 학습할 수 있기 때문에 사용자와 문제 간의 정답 여부 만을 이용하였음에도 좋은 예측 성능을 보였다.  
<br>

## Validation 설정
Test는 사용자가 가장 최근에 푼 문제 하나에 대한 정답 여부를 예측하는 것으로 설계되어있다.  
따라서 Validation도 사용자가 가장 최근에 푼 문제 만을 포함하도록 재구성하였다.  
Test데이터와 개수가 동일하도록 Train 데이터에 포함된 사용자의 10%인 744명의 사용자가 가장 최근에 푼 문제로 Validation을 구성하였다.  
<br>

## Hyperparameter 실험
LightGCN의 Hyperparameter인 n_layers와 hidden_dim에 대한 최적의 값을 찾기 위해 실험을 진행하였다.  
n_layers는 그래프에서 정보가 전파되는 층의 개수를 나타내고, hidden_dim은 사용자와 아이템의 임베딩 크기를 나타낸다.  
Validation의 AUROC를 기준으로 모델의 학습 성능을 확인하였고, 20 epoch 동안 AUROC가 개선되지 않으면 early stopping이 실행되도록 설계하였다.  

| | Train AUC | Train ACC | Valid AUC | Valid ACC | Epoch | Time |
| --- | --- | --- | --- | --- | --- | --- |
| n_layers=1 <br> hidden_dim=64 | 0.8852 | 0.8223 | 0.8150 | 0.7446 | 727 | 1h 21m |
| n_layers=2 | 0.8884 | 0.8254 | 0.8142 | 0.7366 | 1187 | 3h 30m |
| n_layers=3 | 0.8853 | 0.8225 | 0.8117 | 0.7325 | 1695 | 5h 7m |
| hidden_dim=32 | 0.8744 | 0.8118 | 0.8048 | 0.7379 | 946 | 1h 33m |
| hidden_dim=128 | 0.8064 | 0.7681 | 0.7546 | 0.6962 | 42 | 0h 16m |

n_layers의 경우에는, n_layers가 증가함에 따라 소요 시간도 증가하였지만 성능은 오히려 하락하였다. 정보 전파 층이 늘어나면서 연관이 적은 정보도 포함됨에 따라 약간의 오버피팅이 발생하는 것으로 판단된다.  
hidden_dim의 경우에는, 32로 설정했을 때 임베딩의 크기가 작아서 학습 성능이 조금 떨어졌고, 128로 설정했을 때 임베딩의 크기가 너무 커서 학습이 제대로 이루어지지 않았다.  
따라서, n_layers = 1, hidden_dim = 64인 LightGCN 모델을 채택하였다.  
<br>

## Test 결과
| | Public AUC | Public ACC | Private AUC | Private ACC |
| --- | --- | --- | --- | --- |
| LightGCN | 0.7794 | 0.6909 | 0.8145 | 0.7581 |
<br>

## Setup
```bash
(base) conda create -n gcn python=3.10 -y
(base) conda activate gcn
(gcn) pip install -r requirements.txt
```
<br>

## Files
`code/lightgcn`
* `train.py`: 학습코드
* `inference.py`: 추론 후 `submissions.csv` 파일을 만들어주는 소스코드
* `requirements.txt`: 모델 학습에 필요한 라이브러리 목록

`code/lightgcn/lightgcn`
* `args.py`: `argparse`를 통해 학습에 활용되는 여러 argument들 설정
* `datasets.py`: 학습 데이터를 불러 GCN 입력에 맞게 변환
* `trainer.py`: 훈련에 사용되는 함수들
* `utils.py`: 학습에 필요한 부수적인 함수들
<br>

## Run
```bash
(gcn) python train.py --valid_ratio 0.1 --n_epochs 1000 --output_name valid01_epoch1000 --model_name best_model_valid01_epoch1000.pt
(gcn) python inference.py --valid_ratio 0.1 --n_epochs 1000 --output_name valid01_epoch1000 --model_name best_model_valid01_epoch1000.pt
```
<br>
