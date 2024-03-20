import os

import evaluate
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def format_and_camel_case(input_string):
    """
    Convert a hyphen-separated string into camel case.

    Parameters:
    - input_string (str): The input string with hyphens as separators.

    Returns:
    - str: The input string converted to camel case.
    """
    formatted_string = input_string.replace('-', ' ').title()
    formatted_string = formatted_string.replace(' ', '')

    return formatted_string


def tokenize_data(data, tokenizer, key="text", max_length=32):
    """
    Tokenizes the input data using a specified tokenizer.

    Parameters:
    - data (dict): A dictionary containing the input data, with the specified key for text.
    - tokenizer: An instance of a tokenizer from the transformers library.
    - key (str, optional): The key in the input data dictionary that contains the text data (default is "text").
    - max_length (int, optional): Maximum sequence length for tokenization (default is 32).

    Returns:
    - dict: Tokenized data with the same structure as the input data.
    """
    return tokenizer(data[key], max_length=max_length, truncation=True, padding="max_length")


def rename_column(dataset, old_column, new_column):
    """
    Renames a column in the given dataset.

    Parameters:
    - dataset: The dataset to modify.
    - old_column (str): The current name of the column.
    - new_column (str): The new name for the column.

    Returns:
    - Dataset: The modified dataset with the column renamed.
    """
    return dataset.rename_column(old_column, new_column)


def compute_metrics(eval_pred):
    """
    Computes evaluation metrics for a given prediction.

    Parameters:
    - eval_pred (Tuple): A tuple containing model predictions (logits) and true labels.

    Returns:
    - Dict: A dictionary containing evaluation metrics including accuracy, precision, recall, and F1 score.
    """

    # Extract logits and labels
    logits, labels = eval_pred

    # Compute softmax probabilities and predicted labels
    probabilities = tf.nn.softmax(logits)
    predictions = np.argmax(logits, axis=1)

    # Accuracy
    accuracy_metric = evaluate.load("accuracy")
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

    # Precision
    precision_metric = evaluate.load("precision")
    precision = precision_metric.compute(predictions=predictions, references=labels, average='macro')

    # Recall
    recall_metric = evaluate.load("recall")
    recall = recall_metric.compute(predictions=predictions, references=labels, average='macro')

    # F1 Score
    f1_metric = evaluate.load("f1")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')

    # Return metrics in a dictionary
    return {
        "eval_accuracy": accuracy['accuracy'],
        "eval_precision": precision['precision'],
        "eval_recall": recall['recall'],
        "eval_f1": f1['f1']
    }


def read_csv_file(file_path):
    """
    Reads a CSV file using pandas and returns the DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def rename_columns(df, column_mapping):
    """
    Renames columns in a DataFrame based on the provided mapping.

    Parameters:
    - df (pd.DataFrame): The DataFrame to rename columns.
    - column_mapping (dict): A dictionary where keys are existing column names and
      values are the desired new names.

    Returns:
    - pd.DataFrame: The DataFrame with renamed columns.
    """
    df = df.rename(columns=column_mapping)
    return df


def replace_labels(df, label_mapping):
    """
    Replaces labels in a DataFrame based on the provided mapping.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the 'label' column to be replaced.
    - label_mapping (dict): A dictionary where keys are existing labels and values
      are the corresponding replacement values.

    Returns:
    - pd.DataFrame: The DataFrame with replaced labels.
    """
    df['label'] = df['label'].replace(label_mapping)
    return df


def resample_dataset(df, target_column, resampling_method):
    """
    Resample the input DataFrame based on the specified resampling method.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_column (str): The name of the column on which to perform resampling.
    - resampling_method (str): Resampling method, choose from 'oversample', 'undersample', or 'none'.

    Returns:
    pd.DataFrame: Resampled DataFrame with the specified resampling applied.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if resampling_method == 'oversample':
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
    elif resampling_method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y

    resampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target_column)],
                             axis=1)

    return resampled_df


def split_train_test(df, target_column, test_frac):
    """
    Split the input DataFrame into training and testing sets based on the specified target column.
    Ensure that the test set has an equal number of samples for each class.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_column (str): The name of the column on which to perform the split.
    - test_frac (float): Fraction of the total data to include in the test set.

    Returns:
    tuple: A tuple containing train and test DataFrames.
        - train (pd.DataFrame): Training set with features and target values.
        - test (pd.DataFrame): Testing set with features and target values.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    class_counts = y.value_counts()
    min_count = class_counts.min()
    test_size_per_class = int(min_count * test_frac)

    train_dfs = []
    test_dfs = []
    for cls in class_counts.index:
        class_df = df[df[target_column] == cls]
        X_cls = class_df.drop(columns=[target_column])
        y_cls = class_df[target_column]

        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls,
                                                                            test_size=test_size_per_class,
                                                                            random_state=42,
                                                                            stratify=y_cls)
        train_dfs.append(pd.concat([X_train_cls, y_train_cls], axis=1))
        test_dfs.append(pd.concat([X_test_cls, y_test_cls], axis=1))

    # Concatenate the training and test sets
    train = pd.concat(train_dfs)
    test = pd.concat(test_dfs)

    return train, test



def save_to_csv(output_folder, **kwargs):
    """
    Save one or more DataFrames as CSV files in the specified output folder.

    Parameters:
    - output_folder (str): The folder where CSV files will be saved.
    - *args (pd.DataFrame): Variable number of DataFrames to be saved.

    Returns:
    None
    """
    os.makedirs(output_folder, exist_ok=True)

    for df in kwargs.items():
        filename = f'{df[0]}.csv'
        filepath = os.path.join(output_folder, filename)
        df[1].to_csv(filepath, index=False)
