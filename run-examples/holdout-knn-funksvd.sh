#!/bin/bash

# Combination:
#  Rec1 -> item_knn with num_factors=20,lrate=0.01,reg=0.0
#  Rec2 -> FunkSVD with num_factors=20,lrate=0.01,reg=0.0
python3 ../scripts/holdout.py \
    ../Datasets/ml100k/ratings.csv \
    --results_path ../Results/holdout-knn-funksvd.txt \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 item_knn --rec_length 10 \
    --recommender_2 FunkSVD --rec_length 10 \
    --number_iterations 10 \
    --number_positives 40 \
    --number_negatives 120 \
    --number_unlabeled 3000 \
    --params_1 similarity=pearson,k=50,shrinkage=100,normalize=True \
    --params_2 num_factors=20,lrate=0.01,reg=0.0
    #--columns -> Comma separated names for every column.
    #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

# Combination:
#  Rec1 -> item_knn with num_factors=20,lrate=0.01,reg=0.0
#  Rec2 -> FunkSVD with num_factors=20,lrate=0.01,reg=0.0
python3 ../scripts/holdout.py \
    ../Datasets/ml100k/ratings.csv \
    --results_path ../Results/holdout-knn-funksvd.txt \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 item_knn --rec_length 10 \
    --recommender_2 FunkSVD --rec_length 10 \
    --number_iterations 10 \
    --number_positives 40 \
    --number_negatives 120 \
    --number_unlabeled 3000 \
    --params_1 similarity=cosine,k=50,shrinkage=100,normalize=True \
    --params_2 num_factors=20,lrate=0.01,reg=0.0
    #--columns -> Comma separated names for every column.
    #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

# Combination:
#  Rec1 -> item_knn with adj-cosine, k=50 and shrinkage = 100 and normalization
#  Rec2 -> FunkSVD with num_factors=20,lrate=0.01,reg=0.0
python3 ../scripts/holdout.py \
    ../Datasets/ml100k/ratings.csv \
    --results_path ../Results/holdout-knn-funksvd.txt \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 item_knn --rec_length 10 \
    --recommender_2 FunkSVD --rec_length 10 \
    --number_iterations 10 \
    --number_positives 40 \
    --number_negatives 120 \
    --number_unlabeled 3000 \
    --params_1 similarity=adj-cosine,k=50,shrinkage=100,normalize=True \
    --params_2 num_factors=20,lrate=0.01,reg=0.0
    #--columns -> Comma separated names for every column.
    #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.
