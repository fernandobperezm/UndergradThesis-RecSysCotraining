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
#  Rec1 -> item_knn with Pearson, k=50 and shrinkage = 100 and normalization
#  Rec2 -> FunkSVD with num_factors=20,lrate=0.01,reg=0.01
# python3 ../scripts/holdout.py \
#     ../Datasets/ml10m/ratings.csv \
#     --results_path ../Results/knn-bprmf-1/ \
#     --results_file holdout-knn-bprmf-50.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 item_knn --rec_length 10 \
#     --recommender_2 BPRMF --rec_length 10 \
#     --number_iterations 50 \
#     --number_positives $PPOSITIVES \
#     --number_negatives $NNEGATIVES \
#     --number_unlabeled 700000 \
#     --params_1 similarity=pearson,k=5500,shrinkage=500,normalize=True \
#     --params_2 num_factors=20,lrate=0.1,user_reg=0.1,pos_reg=0.001,neg_reg=0.0015,iters=10,sample_with_replacement=True,sampling_type=user_uniform_item_uniform,init_mean=0.0,init_std=0.1,lrate_decay=1.0,rnd_seed=42
#     #--columns -> Comma separated names for every column.
#     #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# Combination:
#  Rec1 -> item_knn with Cosine, k=50 and shrinkage = 100 and normalization
#  Rec2 -> FunkSVD with num_factors=20,lrate=0.01,reg=0.01
# python3 ../scripts/holdout.py \
#     ../Datasets/ml10m/ratings.csv \
#     --results_path ../Results/knn-bprmf-2/ \
#     --results_file holdout-knn-bprmf-50.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 item_knn --rec_length 10 \
#     --recommender_2 BPRMF --rec_length 10 \
#     --number_iterations 50 \
#     --number_positives $PPOSITIVES \
#     --number_negatives $NNEGATIVES \
#     --number_unlabeled 700000 \
#     --params_1 similarity=cosine,k=5500,shrinkage=500,normalize=True \
#     --params_2 num_factors=20,lrate=0.1,user_reg=0.1,pos_reg=0.001,neg_reg=0.0015,iters=10,sample_with_replacement=True,sampling_type=user_uniform_item_uniform,init_mean=0.0,init_std=0.1,lrate_decay=1.0,rnd_seed=42
#     #--columns -> Comma separated names for every column.
#     #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# Combination:
#  Rec1 -> item_knn with Cosine, k=50 and shrinkage = 100 and normalization
#  Rec2 -> BPRMF with num_factors=20,lrate=0.1,user_reg=0.1,pos_reg=0.001,neg_reg=0.0015,iters=10,sample_with_replacement=True,sampling_type=user_uniform_item_uniform,init_mean=0.0,init_std=0.1,lrate_decay=1.0,rnd_seed=42
# python3 ../scripts/holdout.py \
#     ../Datasets/ml10m/ratings.csv \
#     --results_path ../Results/knn-bprmf-3/ \
#     --results_file holdout-knn-bprmf-50.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 item_knn --rec_length 10 \
#     --recommender_2 BPRMF --rec_length 10 \
#     --number_iterations 50 \
#     --number_positives $PPOSITIVES \
#     --number_negatives $NNEGATIVES \
#     --number_unlabeled 700000 \
#     --params_1 similarity=adj-cosine,k=5500,shrinkage=500,normalize=True \
#     --params_2 num_factors=20,lrate=0.1,user_reg=0.1,pos_reg=0.001,neg_reg=0.0015,iters=10,sample_with_replacement=True,sampling_type=user_uniform_item_uniform,init_mean=0.0,init_std=0.1,lrate_decay=1.0,rnd_seed=42
#     #--columns -> Comma separated names for every column.
#     #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

# Combination:
#  Rec1 -> item_knn with Cosine, k=50 and shrinkage = 100 and normalization
#  Rec2 -> BPRMF_THEANO with rank=20,learning_rate = 0.1,lambda_u=0.1,lambda_i=0.1,lambda_j=0.001,lambda_bias=0.0
python3 ../scripts/holdout.py \
    ../Datasets/ml10m/ratings.csv \
    --results_path ../Results/knn-bprmf-3/ \
    --results_file holdout-knn-bprmf_theano-50.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_2 item_knn --rec_length 10 \
    --recommender_1 BPRMF_THEANO --rec_length 10 \
    --number_iterations 50 \
    --number_positives $PPOSITIVES \
    --number_negatives $NNEGATIVES \
    --number_unlabeled $UNLABELED \
    --params_2 similarity=adj-cosine,k=350,shrinkage=0,normalize=True,sparse_weights=True \
    --params_1 rank=20,n_users=69878,n_items=10677,learning_rate=0.1,lambda_u=0.1,lambda_i=0.1,lambda_j=0.001,lambda_bias=0.0 #\
    # --is_binary --make_binary --binary_th 4.0 #\ -> If the dataset is binary.
    # --recover_cotraining --recover_iter 10
    # --params_1 rank=20,n_users=943,n_items=1682,learning_rate=0.1,lambda_u=0.1,lambda_i=0.1,lambda_j=0.001,lambda_bias=0.0
    #--columns -> Comma separated names for every column.
