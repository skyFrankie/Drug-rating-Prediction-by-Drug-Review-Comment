import numpy as np

from config import OUTPUT_PATH
from train_model import load_for_prediction
from utils import (
    clean_n_transform_condition,
    clean_n_transform_drugname,
    clean_n_transform_side_effects,
    clean_review_comment,
    read_data,
    transform_review_comment,
    transform_useful_count,
)



if __name__ == "__main__":
    train_size, train_set, test = read_data()
    (
        cond_dict,
        drug_dict,
        tfidf,
        svd,
        minmax_scaler,
        classifier_list,
    ) = load_for_prediction()
    backup = test.copy()

    test = clean_n_transform_condition(test, cond_dict)
    test = clean_n_transform_drugname(test, drug_dict)
    test = clean_n_transform_side_effects(test)
    test = transform_useful_count(test, minmax_scaler)
    test = clean_review_comment(test)
    test_text_features = transform_review_comment(test, tfidf, svd)

    test_text = test_text_features
    test_eff = test.sideEffects
    test_drug = test.drugName
    test_cond = test.condition
    test_useful = test.usefulCount
    test_features = [test_text, test_eff, test_cond, test_useful, test_drug]

    output_list = [classifier_list[i].predict(test_features) for i in range(5)]
    backup.rating = np.concatenate(
        (
            output_list[0],
            output_list[1],
            output_list[2],
            output_list[3],
            output_list[4],
        ),
        axis=1,
    ).argmax(axis=1)+1
    backup.to_csv(OUTPUT_PATH / r"predicted_testing_data.csv", index=False)
