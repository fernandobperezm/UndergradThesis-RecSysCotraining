#!/bin/bash

# Combination:
#  Rec1 -> item_knn with Pearson, k=50 and shrinkage = 100 and normalization
#  Rec2 -> SLIM_mt with l2_penalty=0.1,l1_penalty=0.001
python3 ../scripts/holdout.py \
    ../Datasets/ml100k/ratings.csv \
    --results_path ../Results/knn-slim-1/ \
    --results_file holdout-knn-slim-50.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 item_knn --rec_length 10 \
    --recommender_2 SLIM_mt --rec_length 10 \
    --number_iterations 50 \
    --number_positives 30 \
    --number_negatives 120 \
    --number_unlabeled 750 \
    --params_1 similarity=pearson,k=50,shrinkage=100,normalize=True \
    --params_2 l2_penalty=0.1,l1_penalty=0.001
    #--columns -> Comma separated names for every column.
    #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# Combination:
#  Rec1 -> item_knn with Cosine, k=50 and shrinkage = 100 and normalization
#  Rec2 -> SLIM_mt with Pearson, k=50 and shrinkage = 100 and normalization
python3 ../scripts/holdout.py \
    ../Datasets/ml100k/ratings.csv \
    --results_path ../Results/knn-slim-2/ \
    --results_file holdout-knn-slim-50.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 item_knn --rec_length 10 \
    --recommender_2 SLIM_mt --rec_length 10 \
    --number_iterations 50 \
    --number_positives 30 \
    --number_negatives 120 \
    --number_unlabeled 750 \
    --params_1 similarity=cosine,k=50,shrinkage=100,normalize=True \
    --params_2 l2_penalty=0.1,l1_penalty=0.001
    #--columns -> Comma separated names for every column.
    #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# Combination:
#  Rec1 -> item_knn with Cosine, k=50 and shrinkage = 100 and normalization
#  Rec2 -> SLIM_mt with Pearson, k=50 and shrinkage = 100 and normalization
python3 ../scripts/holdout.py \
    ../Datasets/ml100k/ratings.csv \
    --results_path ../Results/knn-slim-3/ \
    --results_file holdout-knn-slim-50.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 item_knn --rec_length 10 \
    --recommender_2 SLIM_mt --rec_length 10 \
    --number_iterations 50 \
    --number_positives 30 \
    --number_negatives 120 \
    --number_unlabeled 750 \
    --params_1 similarity=adj-cosine,k=50,shrinkage=100,normalize=True \
    --params_2 l2_penalty=0.1,l1_penalty=0.001
    #--columns -> Comma separated names for every column.
    #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.
