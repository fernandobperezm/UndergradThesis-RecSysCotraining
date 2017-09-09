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

# ItemKNN possible similarity: pearson, cosine, adj-cosine
# Combination:
#  Rec1 -> item_knn with adj-cosine, k=350, shrinkage = 0 and normalization
#  Rec2 -> SLIM_BPR with lambda_i=0.0025,lambda_j=0.00025,learning_rate=0.05,topK=2000
python3 ../../../scripts/holdout.py \
    ../../../Datasets/ml100k/ratings.csv \
    --results_path ../../../Results/knn-slimbpr-3/ \
    --results_file holdout-knn-slimbpr-50.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 item_knn --rec_length 10 \
    --recommender_2 SLIM_BPR --rec_length 10 \
    --number_iterations 50 \
    --number_positives $PPOSITIVES \
    --number_negatives $NNEGATIVES \
    --number_unlabeled $UNLABELED \
    --params_1 similarity=adj-cosine,k=350,shrinkage=0,normalize=True \
    --params_2 lambda_i=0.0025,lambda_j=0.00025,learning_rate=0.05,topK=2000 \
    --make_pop_bins
    # --recover_cotraining --recover_iter 10
    # --is_binary --make_binary --binary_th 4.0 #\ -> If the dataset is binary.
    #--columns -> Comma separated names for every column.
