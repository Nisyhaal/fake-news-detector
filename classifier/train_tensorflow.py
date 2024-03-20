import evaluate
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

model_name = "bert-base-cased"
tokenizer_name = "bert-base-cased"
num_epochs = 5
num_classes = 2
model_directory = f'models/Tensorflow-BERT'

train_data = pd.read_csv(f'dataset/news_6k_train.csv')
test_data = pd.read_csv(f'dataset/news_6k_test.csv')

hg_train_data = Dataset.from_pandas(train_data)
hg_test_data = Dataset.from_pandas(test_data)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


def tokenize_dataset(data):
    return tokenizer(data["text"], max_length=16, truncation=True, padding="max_length")


dataset_train = hg_train_data.map(tokenize_dataset)
dataset_test = hg_test_data.map(tokenize_dataset)

base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

training_args = TrainingArguments(
    output_dir="./results/",
    logging_dir='./results/logs',
    logging_strategy='epoch',
    logging_steps=100,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-6,
    seed=42,
    save_strategy='epoch',
    save_steps=100,
    evaluation_strategy='epoch',
    eval_steps=100,
    load_best_model_at_end=True
)


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
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

model.train()

tokenizer.save_pretrained(model_directory)
model.save_model(model_directory)
