#usr/bin/

python3 ../scripts/k-fold-cotraining.py \
    ../Datasets/ml100k/ratings.csv \
    --holdout_perc 0.8 \
    --header 0 --sep , \
    --user_key user_id --item_key item_id --rating_key rating \
    --rnd_seed 1234 \
    --recommender_1 top_pop --rec_length 10 \
    --recommender_2 top_pop --rec_length 10 \
    --k_fold 2 \
    --number_iterations 30 \
    --number_positives 1 \
    --number_negatives 3 \
    --number_unlabeled 75 \
    #--columns -> Comma separated names for every column.
    #--params_1 -> Params of the recommender 1.
    #--params_2 -> Params of the recommender 2.
    #--is_binary --make_binary --binary_th 4.0 \ -> If the dataset is binary.
