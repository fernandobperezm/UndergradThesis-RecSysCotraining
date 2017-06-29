#!/bin/bash

python3 ../scripts/holdout.py \
    ../Datasets/ml100k/ratings.csv \
    --results_path ../Results/holdout-slim-slim.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 SLIM_mt --rec_length 10 \
    --recommender_2 SLIM_mt --rec_length 10 \
    --k_fold 2 \
    --number_iterations 50 \
    --number_positives 40 \
    --number_negatives 120 \
    --number_unlabeled 3000 \
    --params_1 l2_penalty=0.1,l1_penalty=0.001 \
    --params_2 l2_penalty=0.05,l1_penalty=0.001
    #--columns -> Comma separated names for every column.
    #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.
