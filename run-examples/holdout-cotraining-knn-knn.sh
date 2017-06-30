#!/bin/bash

# Combination:
#  Rec1 -> top_pop
#  Rec2 -> Top_pop
# python3 ../scripts/k-fold-cotraining.py \
#     ../Datasets/ml100k/ratings.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 top_pop --rec_length 10 \
#     --recommender_2 top_pop --rec_length 10 \
#     --k_fold 2 \
#     --number_iterations 30 \
#     --number_positives 40 \
#     --number_negatives 120 \
#     --number_unlabeled 3000 \
#     #--columns -> Comma separated names for every column.
#     #--params_1 -> Params of the recommender 1.
#     #--params_2 -> Params of the recommender 2.
#     #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

# Combination:
#  Rec1 -> item_knn with Pearson, k=50 and shrinkage = 100 and normalization
#  Rec2 -> user_knn with Pearson, k=50 and shrinkage = 100 and normalization
python3 ../scripts/holdout.py \
    ../Datasets/ml100k/ratings.csv \
    --results_path ../Results/knn-knn-1/ \
    --results_file holdout-knn-knn-50.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 item_knn --rec_length 10 \
    --recommender_2 user_knn --rec_length 10 \
    --number_iterations 50 \
    --number_positives 40 \
    --number_negatives 120 \
    --number_unlabeled 3000 \
    --params_1 similarity=pearson,k=50,shrinkage=100,normalize=True \
    --params_2 similarity=pearson,k=50,shrinkage=100,normalize=True
    #--columns -> Comma separated names for every column.
    #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

# # Combination:
# #  Rec1 -> item_knn with Pearson, k=50 and shrinkage = 100 and normalization
# #  Rec2 -> user_knn with Pearson, k=50 and shrinkage = 100 and normalization
# python3 ../scripts/holdout.py \
#     ../Datasets/ml100k/ratings.csv \
#     --results_path ../Results/knn-knn-1/ \
#     --results_file holdout-knn-knn-30.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 item_knn --rec_length 10 \
#     --recommender_2 user_knn --rec_length 10 \
#     --number_iterations 30 \
#     --number_positives 40 \
#     --number_negatives 120 \
#     --number_unlabeled 3000 \
#     --params_1 similarity=pearson,k=50,shrinkage=100,normalize=True \
#     --params_2 similarity=pearson,k=50,shrinkage=100,normalize=True
#     #--columns -> Comma separated names for every column.
#     #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.
#
# Combination:
#  Rec1 -> item_knn with Pearson, k=50 and shrinkage = 100 and normalization
#  Rec2 -> user_knn with Pearson, k=50 and shrinkage = 100 and normalization
# python3 ../scripts/holdout.py \
#     ../Datasets/ml100k/ratings.csv \
#     --results_path ../Results/knn-knn-1/ \
#     --results_file holdout-knn-knn-10.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 item_knn --rec_length 10 \
#     --recommender_2 user_knn --rec_length 10 \
#     --number_iterations 10 \
#     --number_positives 40 \
#     --number_negatives 120 \
#     --number_unlabeled 3000 \
#     --params_1 similarity=pearson,k=50,shrinkage=100,normalize=True \
#     --params_2 similarity=pearson,k=50,shrinkage=100,normalize=True
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
#  Rec2 -> user_knn with Pearson, k=50 and shrinkage = 100 and normalization
python3 ../scripts/holdout.py \
    ../Datasets/ml100k/ratings.csv \
    --results_path ../Results/knn-knn-2/ \
    --results_file holdout-knn-knn-50.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 item_knn --rec_length 10 \
    --recommender_2 user_knn --rec_length 10 \
    --number_iterations 50 \
    --number_positives 40 \
    --number_negatives 120 \
    --number_unlabeled 3000 \
    --params_1 similarity=cosine,k=50,shrinkage=100,normalize=True \
    --params_2 similarity=pearson,k=50,shrinkage=100,normalize=True
    #--columns -> Comma separated names for every column.
    #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

# # Combination:
# #  Rec1 -> item_knn with Cosine, k=50 and shrinkage = 100 and normalization
# #  Rec2 -> user_knn with Pearson, k=50 and shrinkage = 100 and normalization
# python3 ../scripts/holdout.py \
#     ../Datasets/ml100k/ratings.csv \
#     --results_path ../Results/knn-knn-2/ \
#     --results_file holdout-knn-knn-30.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 item_knn --rec_length 10 \
#     --recommender_2 user_knn --rec_length 10 \
#     --number_iterations 30 \
#     --number_positives 40 \
#     --number_negatives 120 \
#     --number_unlabeled 3000 \
#     --params_1 similarity=cosine,k=50,shrinkage=100,normalize=True \
#     --params_2 similarity=pearson,k=50,shrinkage=100,normalize=True
#     #--columns -> Comma separated names for every column.
#     #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.
#
# Combination:
#  Rec1 -> item_knn with Cosine, k=50 and shrinkage = 100 and normalization
#  Rec2 -> user_knn with Pearson, k=50 and shrinkage = 100 and normalization
# python3 ../scripts/holdout.py \
#     ../Datasets/ml100k/ratings.csv \
#     --results_path ../Results/knn-knn-2/ \
#     --results_file holdout-knn-knn-10.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 item_knn --rec_length 10 \
#     --recommender_2 user_knn --rec_length 10 \
#     --number_iterations 10 \
#     --number_positives 40 \
#     --number_negatives 120 \
#     --number_unlabeled 3000 \
#     --params_1 similarity=cosine,k=50,shrinkage=100,normalize=True \
#     --params_2 similarity=pearson,k=50,shrinkage=100,normalize=True
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
#  Rec2 -> user_knn with Pearson, k=50 and shrinkage = 100 and normalization
python3 ../scripts/holdout.py \
    ../Datasets/ml100k/ratings.csv \
    --results_path ../Results/knn-knn-3/ \
    --results_file holdout-knn-knn-50.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 item_knn --rec_length 10 \
    --recommender_2 user_knn --rec_length 10 \
    --number_iterations 50 \
    --number_positives 40 \
    --number_negatives 120 \
    --number_unlabeled 3000 \
    --params_1 similarity=adj-cosine,k=50,shrinkage=100,normalize=True \
    --params_2 similarity=pearson,k=50,shrinkage=100,normalize=True
    #--columns -> Comma separated names for every column.
    #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

# # Combination:
# #  Rec1 -> item_knn with adj-cosine, k=50 and shrinkage = 100 and normalization
# #  Rec2 -> user_knn with Pearson, k=50 and shrinkage = 100 and normalization
# python3 ../scripts/holdout.py \
#     ../Datasets/ml100k/ratings.csv \
#     --results_path ../Results/knn-knn-3/ \
#     --results_file holdout-knn-knn-30.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 item_knn --rec_length 10 \
#     --recommender_2 user_knn --rec_length 10 \
#     --number_iterations 30 \
#     --number_positives 40 \
#     --number_negatives 120 \
#     --number_unlabeled 3000 \
#     --params_1 similarity=adj-cosine,k=50,shrinkage=100,normalize=True \
#     --params_2 similarity=pearson,k=50,shrinkage=100,normalize=True
#     #--columns -> Comma separated names for every column.
#     #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.

# # Combination:
# #  Rec1 -> item_knn with Cosine, k=50 and shrinkage = 100 and normalization
# #  Rec2 -> user_knn with Pearson, k=50 and shrinkage = 100 and normalization
# python3 ../scripts/holdout.py \
#     ../Datasets/ml100k/ratings.csv \
#     --results_path ../Results/knn-knn-3/ \
#     --results_file holdout-knn-knn-10.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 item_knn --rec_length 10 \
#     --recommender_2 user_knn --rec_length 10 \
#     --number_iterations 10 \
#     --number_positives 40 \
#     --number_negatives 120 \
#     --number_unlabeled 3000 \
#     --params_1 similarity=adj-cosine,k=50,shrinkage=100,normalize=True \
#     --params_2 similarity=pearson,k=50,shrinkage=100,normalize=True
#     #--columns -> Comma separated names for every column.
#     #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.
