# FM, FFM

## Setup
```bash
cd /opt/ml/input/code/FM
pip install -r requirements.txt
python main.py --model FM
```

## Files
`code/FM`
* `main.py`: 실행시키면 모델이 학습됩니다.
* `train.py`: 모델을 학습하는 소스코드를 포함합니다
* `utils.py`: 학습에 필요한 부수적인 함수들을 포함합니다.
* `data_process.py`: 데이터 전처리 코드들을 포함합니다.
* `requirements.txt`: 모델 학습에 필요한 라이브러리들이 정리되어 있습니다.

`code/FM/models`
* `FM.py`: FM 모델 코드입니다.
* `FFM.py`: FFM 모델 코드입니다.
