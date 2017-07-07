# #!/bin/bash
#
# # # Combination:
# # #  Rec1 -> FunkSVD with number of factors = 20,learning rate = 0.01 and regularization = 0.0
# # #  Rec2 -> AsySVD with number of factors = 20,learning rate = 0.01, regularization = 0.015, and 10 iterations.
# python3 ../scripts/holdout.py \
#     ../Datasets/ml100k/ratings.csv \
#     --results_path ../Results/mf-mf-1/ \
#     --results_file holdout-mf-mf-50.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 FunkSVD --rec_length 10 \
#     --recommender_2 AsySVD --rec_length 10 \
#     --number_iterations 50 \
#     --number_positives 30 \
#     --number_negatives 120 \
#     --number_unlabeled 750 \
#     --params_1 num_factors=20,lrate=0.01,reg=0.010,iters=10 \
#     --params_2 num_factors=20,lrate=0.01,reg=0.100,iters=10
#     #--columns -> Comma separated names for every column.
#     #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.
#
# # Combination:
# #  Rec1 -> BPRMF with num_factors=20,lrate=0.1,user_reg=0.1,pos_reg=0.001,neg_reg=0.0015,iters=10,
# #                     sample_with_replacement=True,sampling_type=user_uniform_item_pop,sampling_pop_alpha=0.75,
# #                     init_mean=0.0,init_std=0.1,lrate_decay=1.0,rnd_seed=42
# #  Rec2 -> BPRMF with num_factors=20,lrate=0.1,user_reg=0.1,pos_reg=0.001,neg_reg=0.0015,iters=10,
# #                     sample_with_replacement=True,sampling_type=user_uniform_item_uniform,
# #                     init_mean=0.0,init_std=0.1,lrate_decay=1.0,rnd_seed=42
# python3 ../scripts/holdout.py \
#     ../Datasets/ml100k/ratings.csv \
#     --results_path ../Results/mf-mf-2/ \
#     --results_file holdout-mf-mf-50.csv \
#     --holdout_perc 0.8 \
#     --header 0 --sep , \
#     --user_key user_id --item_key item_id --rating_key rating \
#     --rnd_seed 1234 \
#     --recommender_1 BPRMF --rec_length 10 \
#     --recommender_2 BPRMF --rec_length 10 \
#     --number_iterations 50 \
#     --number_positives 30 \
#     --number_negatives 120 \
#     --number_unlabeled 750 \
#     --params_1 num_factors=20,lrate=0.1,user_reg=0.1,pos_reg=0.001,neg_reg=0.0015,iters=10,sample_with_replacement=True,sampling_type=user_uniform_item_pop,sampling_pop_alpha=0.75,init_mean=0.0,init_std=0.1,lrate_decay=1.0,rnd_seed=42 \
#     --params_2 num_factors=20,lrate=0.1,user_reg=0.1,pos_reg=0.001,neg_reg=0.0015,iters=10,sample_with_replacement=True,sampling_type=user_uniform_item_uniform,init_mean=0.0,init_std=0.1,lrate_decay=1.0,rnd_seed=42
#     #--columns -> Comma separated names for every column.
#     # --is_binary --make_binary --binary_th 4.0 \
