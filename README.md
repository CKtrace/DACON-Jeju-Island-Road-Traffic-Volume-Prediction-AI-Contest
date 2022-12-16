# 제주도 도로 교통량 예측 AI 경진대회

```
Final Ranking : 105/712 (Top 14.7%)
```
</br>

## Introduction
</br>

__This repository is a place to share "[제주도 도로 교통량 예측 AI 경진대회](https://dacon.io/competitions/official/236013/overview/description)" solution code.__

</br>

```
주최 : 제주 테크노파크, 제주특별자치도
주관 : 데이콘
```
<br>

## Repository Structure

<br>

```
│  README.md
│  
├─Data_Preprocessing
│       Step1_Preprocessing.ipynb
│       Step2_Preprocessing.ipynb
│   
└─Models
        lgbm.py
        lgbm_optuna.py
```

<br>


## Development Environment
</br>

```
CPU : Intel i9-10900F
GPU : NVIDIA GeForce RTX 3080 Ti
RAM : 32GB
```
</br>

## Approach Method Summary
</br>


```
* Categorical Data more than 2 items    -> LabelEncoding

* Categorical Data 2 items              -> One-Hot Encoding

* Using Best Model in Several Models    -> LGBM

* Finding Best HyperParameter           -> Optuna
```

</br>

## 