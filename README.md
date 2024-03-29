# Deep Knowledge Tracing (DKT)

## OverView
최근 들어서 데이터 사이언스를 이용해 자신의 학습을 상태를 파악하는 서비스가 배포되고 있다.  
Riiid의 산타토익과 같이, 수험자의 문제풀이 데이터를 바탕으로 강점과 약점을 파악해 이를 보완할 다음 문제를 제시하는 방식이다.   
시험을 통해서는 우리 개개인에 맞춤화된 피드백을 받기가 어렵고 따라서 무엇을 해야 성적을 올릴 수 있을지 판단하기 어렵다.   
이럴 때 사용할 수 있는 것이 "지식 상태"를 추적하는 딥러닝 방법론인 DKT이다.  

DKT를 활용하면 우리는 학생 개개인에게 성취도와 취약한 부분을 극복하기 위해 어떤 문제들을 풀면 좋을지 추천이 가능하다.  
대회에서는 학생 개개인의 이해도를 가리키는 지식 상태를 예측하는 일보다는, 주어진 문제를 맞출지 틀릴지 예측하는 것에 집중한다.  

### Task
각 학생이 푼 문제 리스트와 정답 여부가 담긴 데이터를 받아 userID별 마지막 문제를 맞출지 틀릴지 예측
<br><br>

## Component

### 프로젝트 디렉토리 구조 
```
📦level2-dkt-recsys-03
 ┣ 📂EDA
 ┣ 📂data
 ┣ 📂ensemble
 ┣ 📂Feature_Engineering
 ┗ 📂code
   ┣ 📂FM
   ┃ ┣ 📂models
   ┃ ┣ 📂submit
   ┣ 📂dkt
   ┃ ┣ 📂dkt
   ┣ 📂lgbm
   ┃ ┣ 📂outputs
   ┣ 📂lightgcn
   ┃ ┣ 📂lightgcn
   ┗ 📂xgb
```
### 데이터셋 구조

<p align="left"><img src="https://github.com/boostcampaitech6/level2-dkt-recsys-03/assets/43164670/5c9b4041-c426-4ccf-aec0-c6ab664a02f1" width="500"/></p>
<br><br>


## Team
<br>
<table align="left">
  <tr height="155px">
    <td align="center" width="150px">
      <a href="https://github.com/ksb3966"><img src="https://github.com/ksb3966.png" width="100px;" alt=""/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/SiwooPark00"><img src="https://github.com/SiwooPark00.png" width="100px;" alt=""/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/arctic890"><img src="https://github.com/arctic890.png" width="100px;" alt=""/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/JaeGwon-Lee"><img src="https://github.com/JaeGwon-Lee.png" width="100px;" alt=""/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/jinmin111"><img src="https://github.com/jinmin111.png" width="100px;" alt=""/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/chris3427"><img src="https://github.com/chris3427.png" width="100px;" alt=""/></a>
    </td>
  </tr>
  <tr height="80px">
    <td align="center" width="150px">
      <a href="https://github.com/ksb3966">김수빈_T6021</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/SiwooPark00">박시우_T6060</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/arctic890">백승빈_T6075</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/JaeGwon-Lee">이재권_T6131</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/jinmin111">이진민_T6139</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/chris3427">장재원_T6149</a>
    </td>
  </tr>
</table>
&nbsp;
<br>

## Role

| 이름 | 역할 |
| --- | --- |
| 김수빈 | EDA, 데이터 전처리, Bert 실험 및 튜닝, Github Setting, DKT Baseline 튜닝  |
| 박시우 | EDA, 데이터 전처리, feature engineering, LGBM feature 실험, GBDT, GRU 베이스라인 구축 및 실험 |
| 백승빈 | EDA, 데이터 전처리, feature engineering, LQTR 구현 및 튜닝 |
| 이재권 | EDA. 데이터 전처리, feature engineering, LightGCN 실험 |
| 이진민 | EDA, 데이터 전처리, feature engineering, LightGBM feature 실험, FM, FFM 구현 및 실험 |
| 장재원 | EDA, 데이터 전처리, feature engineering, LightGBM 고도화 |
<br>

## Experiment Result

### Single Model Result
|  | Public AUC | Public ACC | Private AUC | Private ACC |
| --- | --- | --- | --- | --- |
| LightGBM | **0.8198** | **0.7554** | 0.8406 | **0.7688** |
| XGBoost | 0.8093 | 0.7366 | **0.8498** | **0.7688** |
| CatBoost | 0.7876 | 0.7285 | 0.8172 | 0.7473 |
| GRU | 0.7381 | 0.6828 | 0.8028 | 0.7392 |
| BERT | 0.7378 | 0.6828 | 0.7698 | 0.7043 |
| LQTR | 0.7476 | 0.6909 | 0.7572 | 0.7151 |
| FFM | 0.7697 | 0.7016 | 0.8240 | 0.7473 |
| LightGCN | 0.7794 | 0.6909 | 0.8145 | 0.7581 |

### Ensemble Result
|  | Public AUC | Public ACC | Private AUC | Private ACC |
| --- | --- | --- | --- | --- |
| Ensemble 1 | 0.8160 | **0.7634** | **0.8475** | **0.7661** |
| Ensemble 2 | **0.8208** | 0.7581 | 0.8401 | 0.7500 |
| Ensemble 3 | 0.8168 | 0.7608 | 0.8435 | 0.7554 |

    Ensemble 1 : LGBM_TTFE(0.5) + XGBoost(0.5)
    
    Ensemble 2 : LGBM_Ensemble(0.5) [TTFE, RemoveOS, RemoveOS] + XGBoost(0.5)
    
    Ensemble 3 : LGBM_TTFE(0.5) + Voting(0.5) [LGBM(0.25) + XGBoost(0.1) + CatBoost(0.05) 
                 + FFM(0.025) + LQTR(0.025) + LightGCN(0.025) + GRU(0.0125) + BERT(0.0125)]

최종적으로 Public AUC 기준 가장 높았던 Ensemble 2, Public ACC 기준 가장 높았던 Ensemble 1 제출
<br><br>

## Wrap-Up Report
[DKT Wrap-up Report - Suggestify.pdf](./DKT.Wrap-up.Report.-.Suggestify.pdf)
<br>
