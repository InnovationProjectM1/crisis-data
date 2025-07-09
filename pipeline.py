from classification_tweet import classification
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

import requests



def get_tweets():
    """
    Fetch tweets from the Crisis API.
    """
    base_url = "https://api.crisis.maxlamenace.duckdns.org/"
    
    cpt = 0
    response = requests.get(base_url + "tweets")
    if response.status_code == 200:
        for tweet in response.json():
            cpt += 1
        tweets = response.json()
        df = pd.DataFrame(tweets)
        return df
    else:
        print("Error fetching tweets:", response.status_code)
        return []




def clean_text(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii') 
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def clean_data(df):
    df = df.drop(columns=["timestamp", "classifier"], errors="ignore")
    df.dropna(subset=['tweet_text'], inplace=True)
    df.drop_duplicates(subset=["tweet_text"], inplace=True)
    df["tweet_text"] = df["tweet_text"].apply(clean_text)

    return df.reset_index(drop=True) 

def post_multiple_classifiers(classified_tweets):
    """
    Post classified tweets to the Crisis API.
    """
    base_url = "https://api.crisis.maxlamenace.duckdns.org/"

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(base_url + "classifiers/multiple", headers=headers, data=json.dumps(classified_tweets))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        raise Exception(f"HTTP error occurred: {e}\nResponse: {response.text}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def chunked_post(data, chunk_size=100):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        try:
            response = post_multiple_classifiers(chunk)
            print(f"✅ Chunk {i // chunk_size + 1} envoyé avec succès.")
        except Exception as e:
            print(f"❌ Erreur lors de l’envoi du chunk {i // chunk_size + 1}: {e}")

def main():

    df = get_tweets()

    tweets = clean_data(df)
    
   
    print("\n--- Exécution de la classification 'class' ---")
    result_class = classification("class",tweets)
    
    print("\n--- Exécution de la classification 'subclass' ---")
    result_subclass = classification("subclass",tweets)
    
    print("\n--- Exécution de la classification 'severity' ---")
    result_severity = classification("severity",tweets)

    json_output = []

    for idx, row in tweets.iterrows():
        print(idx)
        entry = {
            "tweet_id": str(row["tweet_id"]),
            "classified_group": result_class[idx],      
            "classified_sub_group": result_subclass[idx],  
            "difficulty": str(result_severity[idx])        
        }
        json_output.append(entry)

    with open("classification_results.json", "w") as f:
        json.dump(json_output, f, indent=2)
    
    chunked_post(json_output,20)

    print("Fichier classification_results.json créé avec succès.")



if __name__ == "__main__":
    main()
