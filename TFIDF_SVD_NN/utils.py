import pickle
import re
import time
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from cleantext import clean
from sklearn import feature_selection
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from config import DATA_PATH, OUTPUT_PATH, TRANSFORMER_PATH


def read_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1.Read dataset from DATA_PATH
    2.Concatenate train and valid data

    :return: size of train data, concatenated data, test data
    """
    train = pd.read_csv(DATA_PATH / "training.csv")
    valid = pd.read_csv(DATA_PATH / "validation.csv")
    test = pd.read_csv(DATA_PATH / "testing.csv")
    train_size = train.shape[0]
    train_set = pd.concat([train, valid]).reset_index(drop=True)
    return train_size, train_set, test


def clean_n_transform_condition(
    _df: pd.DataFrame, input_dict: dict = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    - Clean the 'condition' column
        1.Removing all non word characters
        2.Replacing the all the wrong information, null value and unseen condition into unknown

    - Transform the condition according to dictionary

    - Create a dictionary for checking with unseen data and encode the existing condition with numbers

    :param _df: target dataframe to be clean
    :param input_dict: if condition dict exist, it will be use to replace condition into unknown if it is not in the dictionary
    :return: return dataframe with cleaned condition column and a dictionary if no dictionary input,
             otherwise return dataframe only
    """
    df = _df.copy()
    df["condition"] = df["condition"].apply(
        lambda x: re.sub(
            r"(\d*<\/span>\s*users\s*found\s*this\s*comment\s*helpful.)", "", str(x)
        )
    )
    df["condition"] = df["condition"].apply(
        lambda x: re.sub("[^\w\s]", "", str(x).lower().strip())
    )
    df["condition"].replace("nan", "unknown", inplace=True)
    df["condition"].replace("", "unknown", inplace=True)
    df["condition"].replace(
        "glioblastoma multi", "glioblastoma multiforme", inplace=True
    )
    if input_dict:
        df["condition"] = df["condition"].apply(
            lambda x: input_dict["unknown"]
            if x not in input_dict.keys()
            else input_dict[x]
        )
        return df
    else:
        cond_dict = {x: i for i, x in enumerate(df["condition"].unique())}
        df["condition"] = df["condition"].apply(lambda x: cond_dict[x])
        return df, cond_dict


def clean_n_transform_drugname(
    _df: pd.DataFrame, input_dict: dict = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    - Clean the 'drugName' column
        1.lowercase and strip the drugName

    - Transform the drugName according to dictionary

    - Create a dictionary for checking with unseen data and encode the existing drugs with numbers

    :param _df: target dataframe to be clean
    :param input_dict: if drug dict exist, it will be use to replace drugName into unknown if it is not in the dictionary
    :return: return dataframe with cleaned drugName column and a dictionary if no dictionary input,
             otherwise return dataframe only
    """
    df = _df.copy()
    df.drugName = df.drugName.apply(lambda x: x.lower().strip())
    if input_dict:
        df["drugName"] = df["drugName"].apply(
            lambda x: input_dict["unknown"]
            if x not in input_dict.keys()
            else input_dict[x]
        )
        return df
    else:
        drug_dict = {x: i for i, x in enumerate(df["drugName"].unique())}
        drug_dict['unknown'] = len(drug_dict)
        df["drugName"] = df["drugName"].apply(lambda x: drug_dict[x])
    return df, drug_dict


def clean_n_transform_side_effects(_df: pd.DataFrame) -> pd.DataFrame:
    """
    - Clean the sideEffects column
        1.lowercase and strip the sideEffects

    - Transform the sideEffects according to dictionary

    :param _df: target dataframe to be clean
    :return: return dataframe with cleaned sideEffects column
    """
    df = _df.copy()
    side_effect_dict = {
        "no side effects": 0,
        "mild side effects": 1,
        "moderate side effects": 2,
        "severe side effects": 3,
        "extremely severe side effects": 4,
    }

    df.sideEffects = df.sideEffects.apply(lambda x: x.lower().strip())
    df["sideEffects"] = df["sideEffects"].apply(lambda x: side_effect_dict[x])
    return df


def transform_useful_count(
    _df: pd.DataFrame, minmax_scaler: object = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, object]]:
    """
    - Transform the usefulCount number by MinMax scaler

    - If no MinMax scaler input, the function will train one

    :param _df: target dataframe to needed to transform the usefulCount
    :param minmax_scaler: Fitted MinMaxScaler from sklearn package
    :return: return dataframe with transformed usefulCount column
    """
    df = _df.copy()
    if minmax_scaler:
        df["usefulCount"] = minmax_scaler.transform(
            np.array(df["usefulCount"]).reshape(-1, 1)
        )
        return df
    else:
        scaler = MinMaxScaler()
        df["usefulCount"] = scaler.fit_transform(
            np.array(df["usefulCount"]).reshape(-1, 1)
        )
        return df, scaler


def clean_review_comment(_df: pd.DataFrame) -> pd.DataFrame:
    """
    - Clean the reviewComment column with cleantext package

    :param _df: target dataframe to be clean
    :return: return dataframe with cleaned reviewComment column
    """
    df = _df.copy()
    df.reviewComment = df.reviewComment.apply(clean)
    return df


def transform_review_comment(
    _df: pd.DataFrame,
    tfidf_trans: object = None,
    svd_trans: object = None,
    p_value: float = 0.55,
) -> Union[np.array, Tuple[np.array, object, object]]:
    """
    - Transforms the cleaned reviewComment column of the input dataframe into a SVD matrix
        1.The reviewComment values will be first feed into tfidf transformer and the output tfidf matrix will be shrink by using
          Chi-square test as filtration in order to save the memory and computation power
        2.The shrinked tfidf_matrix will be feed into SVD transformer giving the SVD matrix output

    :param _df:Dataframe need to extract transformed reviewComment features
    :param tfidf_trans: Fitted TfidfVectorizer from sklearn package
    :param svd_trans: Fitted TruncatedSVD from sklearn package
    :param p_value: Value to control  the filtration by Chi Square Test
    :return: Return SVD matrix, TfidfVectorizer object ,TruncatedSVD object if no transformers input, otherwise return SVD matrix only
    """
    if tfidf_trans and svd_trans:
        # Transformation
        tdidf_vector = tfidf_trans.transform(_df.reviewComment).toarray()
        svd_vector = svd_trans.transform(tdidf_vector)
        return svd_vector
    else:
        # Create tfidf transformer
        tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf.fit(_df.reviewComment)

        # Chi-square test
        # Refers to https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
        y = _df["rating"]
        vocab_list = tfidf.get_feature_names()
        p_value_limit = p_value
        temp_features = pd.DataFrame()
        for vocab in np.unique(y):
            chi2, p = feature_selection.chi2(
                tfidf.transform(_df.reviewComment), y == vocab
            )
            temp_features = temp_features.append(
                pd.DataFrame({"feature": vocab_list, "score": 1 - p, "rating": vocab})
            )
            temp_features = temp_features.sort_values(
                ["rating", "score"], ascending=[True, False]
            )
            temp_features = temp_features[temp_features["score"] > p_value_limit]
        vocabulary = temp_features["feature"].unique().tolist()

        tfidf = TfidfVectorizer(vocabulary=vocabulary)

        text_features = tfidf.fit_transform(_df.reviewComment).toarray()
        print(f"Number of words left: {temp_features['feature'].nunique()}")

        # Create SVD transformer
        start_time = time.time()
        svd = TruncatedSVD(n_components=3000, random_state=42)
        end_time = time.time()
        print(f"Time taken to create the svd matrix :{end_time-start_time} seconds")
        svd_vector = svd.fit_transform(text_features)
        print(f"Total variance explained: {np.sum(svd.explained_variance_ratio_):.2f}")

        return svd_vector, tfidf, svd


def save_or_load_dict(name: str, input_dict: dict = None) -> Optional[dict]:
    """
    Save or load dictionary at OUTPUT_PATH

    :param name: Name of the dictionary needed to be loaded or saved
    :param input_dict: dictionary needed to be saved
    :return: loaded dictionary if no dictionary input
    """
    if input_dict:
        with open(OUTPUT_PATH / f"{name}.pkl", "wb") as file:
            pickle.dump(input_dict, file)
        file.close()
        print(f"{name} saved at {OUTPUT_PATH}")
    else:
        with open(OUTPUT_PATH / f"{name}.pkl", "rb") as file:
            dictionary = pickle.load(file)
        file.close()
        return dictionary


def save_or_load_transformer(name: str, transformer: object = None) -> Optional[object]:
    """
    - Save transformer with input transformer name, if transformer input
    - Load trasformer, if no transformer input
    :param name: Name of transformer
    :param transformer: Target transformer needed to be save
    :return: Return loaded transformer if no transformer input, otherwise save the transformer with target name
    """
    if transformer:
        with open(TRANSFORMER_PATH / f"{name}.p", "wb") as file:
            pickle.dump(transformer, file)
        print(f"{name} saved at {TRANSFORMER_PATH}")
        return None
    else:
        with open(TRANSFORMER_PATH / f"{name}.p", "rb") as file:
            _transformer = pickle.load(file)
        print(f"Transformer {name} loaded")
        return _transformer
