# EE-603 project
This repository contains the code for project of the course EE-603 Machine Learning for signal Processing
 
## Contents
1. [Introduction](#introduction)
2. [Data](#data)
3. [Results](#results)

## Introduction
This project consists of exploring solution of two problems 
1. Audio Classification - Classes considered in this project are - speech,music,both simultaneously
2. Audio tagging- In this problem we have to tag the start and end of speech or music in each audio file.
In this project several of the commonly used machine learning techniques have been used for the aforementioned problems.
The techniques which have been used in this project are :
1. Audio Classification
   i. ANN
   ii. CNN
   iii. GMM (To be uploaded)
   iv. GMM with non negative matrix factorization(To be uploaded) 
2. Audio Tagging 
   i.HMM with GMM used to model emission probabilities.
   ii. RNNs
   iii. RNN with CNNs

## Data
Data considered of 10 second audio files each of which have been strongly labelled. In classification problem only class has been inferred.
In audio tagging problem the output is for the form [{"start_time":start_time,"end_time":end_time, "class":class}]

## Results
1.Classification
| Methods | Accuracy | F1-score|
|---|---|---|
|ANN|TODO|TODO
|CNN|TODO|TODO
|GMM|TODO|TODO
|GMM with NMF|TODO|TODO

2.Tagging

| Methods | Accuracy | F1-score|
|---|---|---|
|RNN|TODO|TODO
|RNN with CNN|TODO|TODO
|HMM|TODO|TODO
