./item_recommendation \
    --training-file="SLIM_BPR_Mono_URM_train.csv" \
    --test-file="SLIM_BPR_Mono_URM_test.csv" \
    --recommender="ItemKNN" \
    --no-id-mapping \
    --measures="AUC prec@10 MAP recall@10 NDCG MRR" \
    --recommender-options="k=300 correlation=Cosine"
