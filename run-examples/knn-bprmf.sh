#!/bin/sh
# Poltecnico di Milano.
# knn-bprmf.sh
# Description: This file runs Co-Training with a combination of BPRMF and
#               ItemKNN recommender.
# Created by: Fernando Benjamín Pérez Maurera.
# Last Modified: 09/09/2017.

# The options are: -p <number> -n <number> -u <number>
# -p represents the number of positive examples to label.
# -n represents the number of negative examples to label.
# -u represents the size of the pool of unlabeled samples.
while getopts p:n:u: option
do
    case "${option}"
        in
        p) PPOSITIVES=${OPTARG};;
        n) NNEGATIVES=${OPTARG};;
        u) UNLABELED=${OPTARG};;
    esac
done

# ItemKNN possible similarity: pearson,cosine,adj-cosine
# Combination:
#  Rec1 -> item_knn with adj-cosine, k=350, shrinkage = 0 and normalization
#  Rec2 -> BPRMF_THEANO with rank=20,n_users=69878,n_items=10677,learning_rate=0.1,lambda_u=0.1,lambda_i=0.1,lambda_j=0.001,lambda_bias=0.0
python3 ../scripts/holdout.py \
    ../Datasets/ml10m/ratings.csv \
    --results_path ../Results/knn-bprmf-3/ \
    --results_file holdout-knn-bprmf_theano-50.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 item_knn --rec_length 10 \
    --recommender_2 BPRMF_THEANO --rec_length 10 \
    --number_iterations 50 \
    --number_positives $PPOSITIVES \
    --number_negatives $NNEGATIVES \
    --number_unlabeled $UNLABELED \
    --params_1 similarity=adj-cosine,k=350,shrinkage=0,normalize=True,sparse_weights=True \
    --params_2 rank=20,n_users=943,n_items=1682,learning_rate=0.1,lambda_u=0.1,lambda_i=0.1,lambda_j=0.001,lambda_bias=0.0 \
    --make_pop_bins
    # --is_binary --make_binary --binary_th 4.0 #\ -> If the dataset is binary.
    # --recover_cotraining --recover_iter 10
    # --params_2 rank=20,n_users=69878,n_items=10677,learning_rate=0.1,lambda_u=0.1,lambda_i=0.1,lambda_j=0.001,lambda_bias=0.0 \
    #--columns -> Comma separated names for every column.
