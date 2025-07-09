from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import re
import contractions
import emoji
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np
import torch
import json
import argparse

def classify_tweet(text,model_path,label_file):
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    with open(label_file) as f:
        label_names = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

   
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=-1).item()
    return label_names[pred]

def get_tweet(i, data):
    if i < 1:
        raise ValueError("Index must be >= 1")
        
    row = data.iloc[i - 1]
    return {
            "tweetID": row["tweet_id"],
            "tweet_text": row["tweet_text"],
    }

def classification(type_of_class,tweets):
    results = []

    if type_of_class == "class":
        
        label_for_model = "./saved_model/RoberTweet_finetune_on_class/label_names.json"
        model_path = "./saved_model/RoberTweet_finetune_on_class"

    elif type_of_class == "subclass":
        
        label_for_model = "./saved_model/RoberTweet_finetune_on_subclass/label_names.json"
        model_path = "./saved_model/RoberTweet_finetune_on_subclass"
    
    elif type_of_class == "severity":
        
        label_for_model = "./saved_model/RoberTweet_finetune_on_severity/label_names.json"
        model_path = "./saved_model/RoberTweet_finetune_on_severity"
    

    for i in tqdm(range(1, len(tweets)+1)):   
        tweet_data = get_tweet(i, tweets)
        classification = classify_tweet(tweet_data["tweet_text"],model_path,label_for_model)
        results.append(classification) 
    
    return results
   


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tweet classifier")
    parser.add_argument("type_of_class", choices=["class", "subclass","severity"], help="Specify whether to use 'class' or 'subclass' or 'severity' model")
    args = parser.parse_args()

    results = classification(args.type_of_class)