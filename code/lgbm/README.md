# Deep Knowledge Tracing : LGBM

## Setup
```bash
cd /opt/ml/input/code/lgbm
conda init
(base) . ~/.bashrc
(base) conda create -n lgbm python=3.10 -y
(base) conda activate lgbm
(lgbm) pip install -r requirements.txt
(lgbm) python train.py
```

## Files
`code/lgbm`
* `args.py` : argparse를 통해 학습에 활용되는 여러 argument들을 받아줍니다.
* `datasets.py` : 데이터셋을 준비합니다. 
* `train.py`: 학습코드입니다.
* `trainer.py`: 훈련에 사용되는 함수들을 포함합니다.
* `tuning.py`: 하이퍼 파라미터를 튜닝합니다.
* `utils.py` : 유틸리티 함수들을 포함합니다.
