import pandas as pd
import re
import contractions
import emoji
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np
import torch
import json


def clean_text(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii') 
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


df = pd.read_csv("./dataset/dataset_CRISIS/dataset1_balanced_sample.csv") #df = pd.read_csv("./dataset/dataset_CRISIS/dataset2.csv")
df.dropna(subset=['tweet-text', 'tweet-class'], inplace=True)
df.drop_duplicates(subset=["tweet-text"], inplace=True)
df["tweet-text"] = df["tweet-text"].apply(clean_text)
df["severity"] = df["severity"].astype(str)


label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["severity"])
label_names = list(label_encoder.classes_)

with open("./saved_model/RoberTweet_finetune_on_severity/label_names.json", "w") as f:
    json.dump(label_names, f)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df[["tweet-text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["tweet-text", "label"]])


model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokenize(batch):
    return tokenizer(batch["tweet-text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
train_dataset = train_dataset.remove_columns(["tweet-text"])
test_dataset = test_dataset.remove_columns(["tweet-text"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_names))


accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
    }


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("./saved_model/RoberTweet_finetune_on_severity")
tokenizer.save_pretrained("./saved_model/RoberTweet_finetune_on_severity")
trainer.evaluate()