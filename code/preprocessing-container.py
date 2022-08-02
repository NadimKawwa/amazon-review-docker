
import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)
columns=['label','text']

if __name__ =='__main__':
    #instantiate parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', 
                        type=float, 
                        default=0.3)
    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    
    #read from input file
    input_data_path = os.path.join("/opt/ml/processing/input", "train_10k.csv")
    
    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path, index_col=0)

    df.dropna(inplace=True, how='any')
    df.drop_duplicates(inplace=True)
    
    #check data ratio
    negative_examples, positive_examples = np.bincount(df["label"])
    print(
        "Data after cleaning: {}, {} positive examples, {} negative examples".format(
            df.shape, positive_examples, negative_examples
        )
    )
    
    #train test split
    split_ratio = args.train_test_split_ratio
    print("Splitting data into train and test sets with ratio {}".format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("label", axis=1), 
        df["label"], 
        test_size=split_ratio, 
        random_state=0
    )
    
    #intantiate preprocess
    preprocess = TfidfVectorizer(
        max_features = 200,
        # strip_accents='ascii', 
        # lowercase=True,
        # analyzer = 'word',
        stop_words='english',
        token_pattern = r'(?u)\b\w\w+\b',
        #max_df=0.95,
        #min_df = 5
    )
    preprocess.fit(X_train['text'])
    
    #run the preprocessing job
    print("Running preprocessing and feature engineering transformations")
    train_features = preprocess.fit_transform(X_train['text'])
    test_features = preprocess.transform(X_test['text'])
    
    #convert to dataframe
    train_features_df = pd.DataFrame(train_features.toarray(), columns=preprocess.get_feature_names())
    test_features_df = pd.DataFrame(test_features.toarray(), columns=preprocess.get_feature_names())
    
    print("Train data shape after preprocessing: {}".format(train_features_df.shape))
    print("Test data shape after preprocessing: {}".format(test_features_df.shape))

    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")

    test_features_output_path = os.path.join("/opt/ml/processing/test", "test_features.csv")
    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")
    
    #drop header from dfs
    print("Saving training features to {}".format(train_features_output_path))
    train_features_df.to_csv(train_features_output_path, header=True, index=False)

    print("Saving test features to {}".format(test_features_output_path))
    test_features_df.to_csv(test_features_output_path, header=True, index=False)

    print("Saving training labels to {}".format(train_labels_output_path))
    y_train.to_csv(train_labels_output_path, header=True, index=False)

    print("Saving test labels to {}".format(test_labels_output_path))
    y_test.to_csv(test_labels_output_path, header=True, index=False)
    
    
