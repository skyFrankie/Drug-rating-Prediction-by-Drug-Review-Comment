import warnings
from typing import Tuple

import imblearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    MaxPooling1D,
    Reshape,
)
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from config import OUTPUT_PATH
from utils import (
    clean_n_transform_condition,
    clean_n_transform_drugname,
    clean_n_transform_side_effects,
    clean_review_comment,
    read_data,
    save_or_load_dict,
    save_or_load_transformer,
    transform_review_comment,
    transform_useful_count,
)

warnings.filterwarnings("ignore")
tf.random.set_seed(89)


# Refers to https://towardsdatascience.com/f-beta-score-in-keras-part-i-86ad190a252f
def f1_metric(
    beta: int = 1,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> float:
    def binary_fbeta(
    ytrue: np.array,
    ypred: np.array,
    )-> float:
        """
        A F1-score calculating metric function used for binary classification during training
        :param ytrue: True labels array
        :param ypred: Prediction labels array
        :param beta:  Set to 1 as it is F1-score
        :param threshold: threshold for predicted probability to classify as 1
        :param epsilon: a very small value to avoid the divider is zero
        :return: F1 score
        """
        # squaring beta
        beta_squared = beta**2

        # casting ytrue and ypred as float dtype
        ytrue = tf.cast(ytrue, tf.float32)
        ypred = tf.cast(ypred, tf.float32)

        # setting values of ypred greater than the set threshold to 1 while those lesser to 0
        ypred = tf.cast(tf.greater_equal(ypred, tf.constant(threshold)), tf.float32)

        tp = tf.reduce_sum(ytrue * ypred)  # calculating true positives
        predicted_positive = tf.reduce_sum(ypred)  # calculating predicted positives
        actual_positive = tf.reduce_sum(ytrue)  # calculating actual positives

        # epsilon is set so as to avoid division by zero error
        precision = tp / (predicted_positive + epsilon)  # calculating precision
        recall = tp / (actual_positive + epsilon)  # calculating recall

        # calculating fbeta
        fb = (
            (1 + beta_squared)
            * precision
            * recall
            / (beta_squared * precision + recall + epsilon)
        )
        return fb
    return binary_fbeta

# Define Model
def build_classifier(
    num_cond: int,
    num_drug: int,
    classify_num: int,
    num_eff: int = 5,
    num_vocab: int = 3000,
    dim: int = 50,
    lr: float = 0.000001,
    reduction: str = "auto",
    regulazier_val: float = 0.04,
    nn1: int = 64,
    nn2: int = 64,
    nn3: int = 16,
    drop_out1: float = 0.5,
    drop_out2: float = 0.4,
    metric: object = "binary_accuracy",
) -> object:
    """
    :param num_cond: Number of dimensions input from condition features
    :param num_drug: Number of dimensions input from drugName features
    :param num_eff: Number of dimensions input from sideEffects features
    :param classify_num: Name of class the classifier corresponds to
    :param num_vocab: Number of dimensions input from Text features
    :param dim: Numbers of dimensions the drugName and condition features will be breakdown to in the Embedding layers
    :param lr: Learning rate of the model
    :param reduction: mode to be set in for the loss function
    :param regulazier_val: value set for the l2_regulazier
    :param nn1: Number of filters will be used for the 1DConv layers
    :param nn2: Number of neurons used after the concatenation layer
    :param nn3: Number of neurons used before output
    :param drop_out1: Drop out rate for the nn1
    :param drop_out2: Drop out rate for the nn2
    :param metric: Metric used to monitor the performance of model
    :return: Return the Classifier
    """
    input_text = Input(shape=(num_vocab,))
    reshape = Reshape(target_shape=(6, int(num_vocab / 6)))(input_text)
    stacks = []
    for kernel_size in [2, 3, 4]:
        conv = Conv1D(nn1, kernel_size, padding="same", activation="relu", strides=1)(
            reshape
        )
        pool = MaxPooling1D(pool_size=3)(conv)
        drop = Dropout(0.5)(pool)
        stacks.append(drop)
    stacked = Concatenate()(stacks)
    flat1 = Flatten()(stacked)

    input_drug = Input(shape=(1,))
    drug = Embedding(input_dim=num_drug, output_dim=dim)(input_drug)
    flat_drug = Flatten()(drug)

    input_cond = Input(shape=(1,))
    cond = Embedding(input_dim=num_cond, output_dim=dim)(input_cond)
    flat_cond = Flatten()(cond)

    input_eff = Input(shape=(1,))
    eff = Embedding(input_dim=num_eff, output_dim=3)(input_eff)
    flat_eff = Flatten()(eff)

    input_useful = Input(shape=(1,))
    useful = Dense(1, activation="relu")(input_useful)

    merged = Concatenate()([flat_drug, flat_cond, flat_eff, useful, flat1])
    dense1 = Dense(
        nn2, activation="relu", kernel_regularizer=regularizers.L2(regulazier_val)
    )(merged)
    drop1 = Dropout(drop_out1)(dense1)
    dense2 = Dense(
        nn3, activation="relu", kernel_regularizer=regularizers.L2(regulazier_val)
    )(drop1)
    drop2 = Dropout(drop_out2)(dense2)
    output = Dense(1, activation="sigmoid")(drop2)

    classifier = Model(
        inputs=[input_text, input_eff, input_cond, input_useful, input_drug],
        outputs=output,
    )
    opt = Adam(learning_rate=lr)
    classifier.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=reduction),
        metrics=[metric],
    )
    print(classifier.summary())
    plot_model(
        classifier,
        to_file=f"model_plot_classifier_{classify_num}.png",
        show_shapes=True,
        show_layer_names=True,
    )
    return classifier


# Define Train Model function
def train_model(
    num_classifer: int,
    _labels: pd.Series,
    train_features: list,
    valid_features: list,
    **kwargs,
) -> Tuple[object, object]:
    """
    Train classifier with specific label set
    :param num_classifer: Name of Label the classifier is specified to
    :param _labels: The label set correspond to target label class
    :param train_features: Training set input
    :param train_features: Validation set input
    :param kwargs: Parameters specified to the classifier model
    :return: Return Trained model and training history
    """
    oversample = imblearn.over_sampling.SMOTE(random_state=42)
    train_labels = _labels[:train_size]
    valid_labels = _labels[train_size:]

    # Create Oversampling features
    X0, y0 = oversample.fit_resample(train_features[0], train_labels)
    X1, y1 = oversample.fit_resample(
        np.array(train_features[1]).reshape(-1, 1), train_labels
    )
    X2, y2 = oversample.fit_resample(
        np.array(train_features[2]).reshape(-1, 1), train_labels
    )
    X3, y3 = oversample.fit_resample(
        np.array(train_features[3]).reshape(-1, 1), train_labels
    )
    X4, y4 = oversample.fit_resample(
        np.array(train_features[4]).reshape(-1, 1), train_labels
    )
    tran_train_feature = [X0, X1, X2, X3, X4]
    tran_train_labels = y0

    # Set early stopping and model saving
    es = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)
    mcp_save = ModelCheckpoint(
        OUTPUT_PATH / f"classifier_{num_classifer}.h5",
        save_best_only=True,
        monitor="val_binary_fbeta",
        mode="max",
    )
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(tran_train_labels), y=tran_train_labels
    )
    if kwargs:
        classifier = build_classifier(*(model_const_input + [num_classifer]), **kwargs)
    else:
        classifier = build_classifier(*(model_const_input + [num_classifer]))
    print(f"Now training classifier{num_classifer}...")
    history = classifier.fit(
        tran_train_feature,
        tran_train_labels,
        class_weight={j: x for j, x in enumerate(class_weights)},
        batch_size=128,
        epochs=1000,
        validation_data=(valid_features, valid_labels),
        callbacks=[es, mcp_save],
        verbose=0,
    )
    y_pred = np.where(classifier.predict(valid_features) >= 0.5, 1, 0).reshape(-1)
    y_true = valid_labels
    print(
        f'Micro f1 score for classifier_{num_classifer} : {f1_score(y_true, y_pred, average="micro"):.2f}\n'
    )
    print(
        f'Macro f1 score for classifier_{num_classifer} : {f1_score(y_true, y_pred, average="macro"):.2f}\n'
    )
    print(classification_report(y_true, y_pred, labels=[0, 1]))
    find_opt_f1_score(y_true, classifier.predict(valid_features).reshape(-1))
    return classifier, history


# Define function to plot model training detail
def plot_model_performance(_history: object) -> None:
    """
    Plot the training loss and F1 score of during the model training
    :param _history: Training history of the model
    :return: None
    """
    # Training F1 score
    loss_train = _history.history["binary_fbeta"]
    loss_val = _history.history["val_binary_fbeta"]
    epochs = range(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, "g", label="Training F1")
    plt.plot(epochs, loss_val, "b", label="validation F1")
    plt.title("Training and Validation F1 score")
    plt.xlabel("Epochs")
    plt.ylabel("F1-score")
    plt.legend()
    plt.show()

    # Training loss
    loss_train = _history.history["loss"]
    loss_val = _history.history["val_loss"]
    epochs = range(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, "g", label="Training loss")
    plt.plot(epochs, loss_val, "b", label="validation loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return None


# Definge function for loading model
def load_for_prediction() -> Tuple[dict, dict, object, object, object, list]:
    """
    Load all the necessary dictionary, transformer and model for prediction
    :return: condition_dict, drugName_dict, Tfidf, SVD, MinMax_scaler, list of classifiers
    """
    cond_dict = save_or_load_dict("condition_dict")
    drug_dict = save_or_load_dict("drugName_dict")
    tfidf = save_or_load_transformer("Tfidf")
    svd = save_or_load_transformer("SVD")
    minmax_scaler = save_or_load_transformer("MinMax_scaler")
    classifier_list = []
    for num_classifer in range(5):
        param = save_or_load_dict(f"param_{num_classifer}")
        classifier = load_model(
            OUTPUT_PATH / f"classifier_{num_classifer}.h5",
            custom_objects={"binary_fbeta": f1_metric(threshold=param['metric_threshold'])},
        )
        classifier_list.append(classifier)
    return cond_dict, drug_dict, tfidf, svd, minmax_scaler, classifier_list

#Refers to https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293
def find_opt_f1_score(y_true,y_pred) -> None:
    """
    Finding the best threshold value for highest F1-score to fight the imbalance dataset
    :param y_true: Array of ground truth
    :param y_pred: Array of predicted probability
    :return: best threshold value
    """
    thresholds = np.arange(0.0, 1.0, 0.0001)
    fscore = np.zeros(shape=(len(thresholds)))
    print('Length of sequence: {}'.format(len(thresholds)))

    # Fit the model
    for index, elem in enumerate(thresholds):
        # Corrected probabilities
        y_pred_prob = (y_pred > elem).astype('int')
        # Calculate the f-score
        fscore[index] = f1_score(y_true, y_pred_prob,average="macro")

    # Find the optimal threshold
    index = np.argmax(fscore)
    thresholdOpt = round(thresholds[index], ndigits=4)
    fscoreOpt = round(fscore[index], ndigits=4)
    print('Best Threshold: {} with F-Score: {}'.format(thresholdOpt, fscoreOpt))
    return None


if __name__ == "__main__":
    # Transform and Extract training data and validation data
    train_size, train_set, test = read_data()

    trans_train_set, cond_dict = clean_n_transform_condition(train_set)
    trans_train_set, drug_dict = clean_n_transform_drugname(trans_train_set)
    trans_train_set = clean_n_transform_side_effects(trans_train_set)
    trans_train_set, minmax_scaler = transform_useful_count(trans_train_set)
    trans_train_set = clean_review_comment(trans_train_set)
    trans_train_set_text_features, tfidf, svd = transform_review_comment(
        trans_train_set
    )

    # Save all the transformer and dict
    save_or_load_dict("condition_dict", cond_dict)
    save_or_load_dict("drugName_dict", drug_dict)
    save_or_load_transformer("MinMax_scaler", minmax_scaler)
    save_or_load_transformer("SVD", svd)
    save_or_load_transformer("Tfidf", tfidf)

    # Prepare for labels
    # Minus one for model input preparation
    # Create 5 sets of labels for one vs all classification
    for label in np.unique(trans_train_set.rating - 1):
        globals()[f"labels_{label}"] = np.where(
            trans_train_set.rating - 1 == label, 1, 0
        )

    # Gathering Train features
    train_text = trans_train_set_text_features[:train_size]
    train_eff = trans_train_set.sideEffects[:train_size]
    train_drug = trans_train_set.drugName[:train_size]
    train_cond = trans_train_set.condition[:train_size]
    train_useful = trans_train_set.usefulCount[:train_size]
    train_features = [train_text, train_eff, train_drug, train_cond, train_useful]

    # Gathering Valid features
    valid_text = trans_train_set_text_features[train_size:]
    valid_eff = trans_train_set.sideEffects[train_size:]
    valid_drug = trans_train_set.drugName[train_size:]
    valid_cond = trans_train_set.condition[train_size:]
    valid_useful = trans_train_set.usefulCount[train_size:]
    valid_features = [valid_text, valid_eff, valid_drug, valid_cond, valid_useful]

    # Model Training
    model_const_input = [len(cond_dict), len(drug_dict)]

    # Define model params
    param_0 = {'metric':f1_metric(),
               'lr':1.48344e-3,
               'nn1': 128}
    param_1 = {'lr': 6.867e-6,
               'metric': f1_metric(threshold=0.38),
               'dim': 50,
               'nn1': 128,
               'nn2': 64,
               'nn3': 16,
               'drop_out1': 0.5}
    param_2 = {"lr": 2.469e-4, "metric": f1_metric(threshold=0.38)}
    param_3 = {
        "lr": 2.2e-4,
        "metric": f1_metric(threshold=0.41),
        "nn1": 128,
        "nn2": 32,
        "nn3": 16,
        "drop_out1": 0.4,
        "drop_out2": 0.5,
    }
    param_4 = {'lr': 3.221e-4,
               'metric': f1_metric(),
               'nn1': 128,
               'nn2': 64,
               'nn3': 16,
               'drop_out1': 0.5,
               'drop_out2': 0.5}
    # Train Classifier
    for i, param in enumerate([param_0, param_1, param_2, param_3, param_4]):
        globals()[f"classifier{i}"], globals()[f"history{i}"] = train_model(
            i, globals()[f"labels_{i}"], train_features, valid_features, **param
        )
        plot_model_performance(globals()[f"history{i}"])
        del param['metric']
        if i == 1:
            param['metric_threshold']=0.38
        elif i == 2:
            param['metric_threshold']=0.38
        elif i == 3:
            param['metric_threshold']=0.42
        else:
            param['metric_threshold']=0.5
        save_or_load_dict(f"param_{i}", param)

    # one vs all performance on validation set
    output_list = [
        globals()[f"classifier{i}"].predict(valid_features) for i in range(5)
    ]
    y_pred = np.concatenate(
        (
            output_list[0],
            output_list[1],
            output_list[2],
            output_list[3],
            output_list[4],
        ),
        axis=1,
    ).argmax(axis=1)
    y_true = trans_train_set.rating[train_size:] - 1
    print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4]))
    print(f'Micro f1 score: {f1_score(y_true, y_pred, average="micro"):.2f}')
    print(f'Macro f1 score: {f1_score(y_true, y_pred, average="macro"):.2f}')
