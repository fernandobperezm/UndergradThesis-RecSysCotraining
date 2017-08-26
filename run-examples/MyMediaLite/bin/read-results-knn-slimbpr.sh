#!/bin/bash

# The options are: -p <number> -n <number>
# -p represents the number of positive examples to label.
# -n represents the number of negative examples to label.
while getopts p:n:u: option
do
    case "${option}"
        in
        p) PPOSITIVES=${OPTARG};;
        n) NNEGATIVES=${OPTARG};;
        u) UNLABELED=${OPTARG};;
    esac
done

# Combination:
#  Rec1 -> item_knn with Cosine, k=50 and shrinkage = 100 and normalization
#  Rec2 -> SLIM with Pearson, k=50 and shrinkage = 100 and normalization
python3 ../../../scripts/read_results.py \
    ../../../Datasets/ml10m/ratings.csv \
    --results_path ../../../Results/knn-slimbpr-3/ \
    --results_file holdout-knn-slim-50.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_2 item_knn --rec_length 10 \
    --recommender_1 SLIM_BPR --rec_length 10 \
    --number_iterations 50 \
    --number_positives $PPOSITIVES \
    --number_negatives $NNEGATIVES \
    --number_unlabeled $UNLABELED \
    --params_2 similarity=adj-cosine,k=350,shrinkage=0,normalize=True \
    --params_1 lambda_i=0.0025,lambda_j=0.00025,learning_rate=0.05,topK=2000 \
    --to_read label_comparison,numberlabeled \
    --make_pop_bins
