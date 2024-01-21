#!/bin/bash
# Written by sbkim8519@gmail.com for ai_tech_6_level_2_DKT competiton

# Data Download
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000268/data/data.tar.gz ~ /data.tar.gz
# Baseline Download

# wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000268/data/code.tar.gz ~ /code.tar.gz

# 압축 파일 해제 및 삭제
tar -zxvf code.tar.gz
# tar -zxvf data.tar.gz
rm -r *.gz

sleep 3
echo "Data 및 Baseline Code 세팅 완료"

# miniconda 설치
MINICONDA_INSTALLER_SCRIPT="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_PREFIX="${HOME}/miniconda3"

# Miniconda 다운로드
wget https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER_SCRIPT} -O /tmp/${MINICONDA_INSTALLER_SCRIPT}

# Miniconda 설치
bash /tmp/${MINICONDA_INSTALLER_SCRIPT} -b -p ${MINICONDA_PREFIX}

# Miniconda 초기화 및 PATH 설정
eval "$(${MINICONDA_PREFIX}/bin/conda shell.bash hook)"
conda init
conda config --set auto_activate_base false

# 임시 파일 삭제
rm /tmp/${MINICONDA_INSTALLER_SCRIPT}

echo "Miniconda 설치가 완료"

CONDA_BASE=$(conda info --base)
. $CONDA_BASE/etc/profile.d/conda.sh

# dkt 가상환경 설정
cd code/dkt
conda create --name dkt python=3.8 -y
conda activate dkt
pip install -r requirements.txt 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
echo "DKT 가상환경 설정 완료"

conda deactivate

# lightgcn 가상환경 설정
cd ..
cd lightgcn
conda create --name lightgcn python=3.8 -y
conda activate lightgcn
pip install -r requirements.txt 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y 
echo "LightGCN 가상환경 설정 완료"