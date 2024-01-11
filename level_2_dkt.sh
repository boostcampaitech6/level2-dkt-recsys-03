#!/bin/bash
# Written by sbkim8519@gmail.com for ai_tech_6_level_2_DKT competiton

# Data Download
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000268/data/data.tar.gz ~ /data.tar.gz
# Baseline Download
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000268/data/code.tar.gz ~ /code.tar.gz

# 압축 파일 해제 및 삭제
tar -zxvf code.tar.gz
tar -zxvf data.tar.gz
rm -r *.gz

sleep 3
echo "Data 및 Baseline Code 세팅 완료"



# dkt 가상환경 설정
cd code/dkt
conda create --name dkt python=3.8 -y
conda activate dkt
pip install -r requirements.txt 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
echo "DKT 가상환경 설정 완료"

# lightgcn 가상환경 설정
cd ..
cd lightgcn
conda create --name lightgcn python=3.8 -y
conda activate lightgcn
pip install -r requirements.txt 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y 
echo "LightGCN 가상환경 설정 완료"