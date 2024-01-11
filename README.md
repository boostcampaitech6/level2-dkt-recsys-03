# Deep Knowledge Tracing (DKT)

## OverVIew
Boostcamp A.I. Tech DKT 트랙 베이스라인 코드입니다.
현재 DKT 대회는 두 종류의 베이스라인이 제공됩니다.
+ `dkt/` 이 폴더 내에는 **Sequential Model**로 풀어나가는 베이스라인이 담겨져있습니다.
+ `lightgcn/` 이 폴더 내에는 Graph 기법으로 풀어나가는 베이스라인이 담겨져있습니다.

두 베이스라인의 파일 구조는 많이 비슷합니다. 그러나 사용하는 라이브러리의 차이가 있기 때문에 **`conda` 환경을 분리해서 사용하는 것을 추천**드립니다.

---
## Component
코드 구조는 아래와 같습니다. 
+ Bert, LSTM 등 Sequential한 접근법을 가진 `dkt/` 폴더와, graph 기법을 사용한 `lightgcn/` 이 있습니다.
+ `level_2_dkt.sh` bash 파일에는 `dkt/` 및 `lightgcn/`을 실행하기 위한 환경 설정 명령어가 있습니다. 


```
├── code
│   ├── README.md
│   ├── __init__.py
│   ├── dkt
│   │   ├── README.md
│   │   ├── dkt
│   │   │   ├── args.py
│   │   │   ├── criterion.py
│   │   │   ├── dataloader.py
│   │   │   ├── metric.py
│   │   │   ├── model.py
│   │   │   ├── optimizer.py
│   │   │   ├── scheduler.py
│   │   │   ├── trainer.py
│   │   │   └── utils.py
│   │   ├── inference.py
│   │   ├── requirements.txt
│   │   └── train.py
│   └── lightgcn
│       ├── README.md
│       ├── inference.py
│       ├── lightgcn
│       │   ├── args.py
│       │   ├── datasets.py
│       │   ├── trainer.py
│       │   └── utils.py
│       ├── requirements.txt
│       └── train.py
├── data
│   ├── sample_submission.csv
│   ├── test_data.csv
│   └── train_data.csv
└── level_2_dkt.sh
```
---
