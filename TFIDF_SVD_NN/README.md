# COMP5434 Project - Drug Rating Prediction With Drug Review

There are two approaches in total to carry out the prediction, both of them used a one vs rest classification technique.

1. Feed in Tfidf-SVD matrix from text, Categorical Features and Numeric feature to 5 Neural networks,each for one class respectively. Simply pick the highest probability one as the predicted class.
2. Same inputs and 5 Neural network approach as first one, use a decision tree model instead of picking the highest probability one as the predicted class.

The code only shows the best approach, which is the first one without decision tree.

Please run the program in the following step to reproduce result:

1. Modify parameters in `config.py`.
   1. Open `config.py`
   2. Change the directory of data source that contains `training.csv`,`validation.csv` and `testing.csv`.
   3. Change `OUTPUT_PATH` as desired output path.
    It will automatically create a transformer folder inside the output folder if no such folder detected.
2. model training.
   1. Run the `train_model.py`. It will use the helper functions in `utils.py` to transform the data to model input format and perform training.
3. model inference.
    1. All the trained transformers, models and dictionary will be saved into the `OUTPUT_PATH`. Run `infer_model.py` to carry out prediction to `testing.csv`.
    2. The prediction result wil be saved as `predicted_testing_data.csv` in the `OUTPUT_PATH`.

NOTE: _The detail EDA of the data can be found in the `EDA.ipynb` jupyter notebook._
