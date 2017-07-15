#!/bin/bash

# The options are: -p <number> -n <number>
# -p represents the number of positive examples to label.
# -n represents the number of negative examples to label.
while getopts p:n: option
do
    case "${option}"
        in
        p) PPOSITIVES=${OPTARG};;
        n) NNEGATIVES=${OPTARG};;
    esac
done

# python3 ../scripts/holdout.py \
#     ../Datasets/ml10m/ratings.csv \
#     --results_path ../Results/funksvd-bprmf-1/ \
#     --results_file holdout-funksvd-bprmf-50.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 FunkSVD --rec_length 10 \
#     --recommender_2 BPRMF --rec_length 10 \
#     --number_iterations 50 \
#     --number_positives $PPOSITIVES \
#     --number_negatives $NNEGATIVES \
#     --number_unlabeled 700000 \
#     --params_1 num_factors=20,lrate=0.01,reg=0.01 \
#     --params_2 num_factors=20,lrate=0.1,user_reg=0.1,pos_reg=0.001,neg_reg=0.0015,iters=10,sample_with_replacement=True,sampling_type=user_uniform_item_uniform,init_mean=0.0,init_std=0.1,lrate_decay=1.0,rnd_seed=42
#     #--columns -> Comma separated names for every column.
#     #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

python3 ../scripts/holdout.py \
    ../Datasets/ml10m/ratings.csv \
    --results_path ../Results/funksvd-bprmf-1/ \
    --results_file holdout-funksvd-bprmf-50.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 FunkSVD --rec_length 10 \
    --recommender_2 BPRMF_THEANO --rec_length 10 \
    --number_iterations 50 \
    --number_positives $PPOSITIVES \
    --number_negatives $NNEGATIVES \
    --number_unlabeled 700000 \
    --params_1 num_factors=20,lrate=0.01,reg=0.01 \
    --params_2 rank=20,n_users=69878,n_items=10677,learning_rate=0.1,lambda_u=0.1,lambda_i=0.1,lambda_j=0.001,lambda_bias=0.0
    #--columns -> Comma separated names for every column.
    #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.
