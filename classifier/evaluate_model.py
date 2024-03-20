import evaluate
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

model_directory = f'models/Tensorflow-BERT'

train_data = pd.read_csv(f'dataset/news_6k_train.csv')
test_data = pd.read_csv(f'dataset/news_6k_test.csv')

tokenizer = AutoTokenizer.from_pretrained(model_directory)
base_model = AutoModelForSequenceClassification.from_pretrained(model_directory)

hg_train_data = Dataset.from_pandas(train_data)
hg_test_data = Dataset.from_pandas(test_data)


def tokenize_dataset(data):
    return tokenizer(data["text"], max_length=16, truncation=True, padding="max_length")


dataset_train = hg_train_data.map(tokenize_dataset)
dataset_test = hg_test_data.map(tokenize_dataset)


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probabilities = tf.nn.softmax(logits)
    predictions = np.argmax(logits, axis=1)

    accuracy_metric = evaluate.load("accuracy")
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

    precision_metric = evaluate.load("precision")
    precision = precision_metric.compute(predictions=predictions, references=labels, average='macro')

    recall_metric = evaluate.load("recall")
    recall = recall_metric.compute(predictions=predictions, references=labels, average='macro')

    f1_metric = evaluate.load("f1")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')

    return {
        "eval_accuracy": accuracy['accuracy'],
        "eval_precision": precision['precision'],
        "eval_recall": recall['recall'],
        "eval_f1": f1['f1']
    }


model = Trainer(
    model=base_model,
    compute_metrics=compute_metrics
)

train_results = model.evaluate(dataset_train)

print("Training Set Evaluation:")
print("Accuracy:", train_results["eval_accuracy"])
print("Precision:", train_results["eval_precision"])
print("Recall:", train_results["eval_recall"])
print("F1:", train_results["eval_f1"])

model = Trainer(
    model=base_model,
    compute_metrics=compute_metrics
)

test_results = model.evaluate(dataset_test)

print("Testing Set Evaluation:")
print("Accuracy:", test_results["eval_accuracy"])
print("Precision:", test_results["eval_precision"])
print("Recall:", test_results["eval_recall"])
print("F1:", test_results["eval_f1"])
