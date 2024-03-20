import pandas as pd
import torch
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler

model_name = "bert-base-cased"
tokenizer_name = "bert-base-cased"
num_epochs = 5
num_classes = 2
model_directory = f'models/Pytorch-BERT'

train_data = pd.read_csv(f'dataset/news_6k_train.csv')
test_data = pd.read_csv(f'dataset/news_6k_test.csv')

hg_train_data = Dataset.from_pandas(train_data)
hg_test_data = Dataset.from_pandas(test_data)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


def tokenize_dataset(data):
    return tokenizer(data["text"], max_length=16, truncation=True, padding="max_length")


dataset_train = hg_train_data.map(tokenize_dataset)
dataset_test = hg_test_data.map(tokenize_dataset)

columns_to_remove = ["text", "__index_level_0__"]

for column in columns_to_remove:
    try:
        dataset_train = dataset_train.remove_columns([column])
        dataset_test = dataset_test.remove_columns([column])
    except ValueError:
        pass

dataset_train = dataset_train.rename_column("label", "labels")
dataset_test = dataset_test.rename_column("label", "labels")

dataset_train.set_format("torch")
dataset_test.set_format("torch")

torch.cuda.empty_cache()

train_dataloader = DataLoader(dataset=dataset_train, shuffle=True, batch_size=16)
eval_dataloader = DataLoader(dataset=dataset_test, batch_size=16)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

num_training_steps = num_epochs * len(train_dataloader)
optimizer = AdamW(params=model.parameters(), lr=5e-6)
lr_scheduler = get_scheduler(name="linear",
                             optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

tokenizer.save_pretrained(model_directory)
model.save_pretrained(model_directory)
