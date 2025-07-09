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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def clean_text(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii') 
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

df = pd.read_csv("./dataset/dataset_CRISIS/dataset2.csv")
df.dropna(subset=['tweet-text', 'tweet-class'], inplace=True)
df.drop_duplicates(subset=["tweet-text"], inplace=True)
df["tweet-text"] = df["tweet-text"].apply(clean_text)
#df["severity"] = df["severity"].astype


with open("./saved_model/RoberTweet_finetune_on_subclass/label_names.json") as f:
    label_names = json.load(f)

# Chargement du modèle et tokenizer fine-tunés
model_path = "./saved_model/RoberTweet_finetune_on_subclass"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify_tweet(text):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=-1).item()
    return label_names[pred]

def get_data(i, data):
    if i < 1:
        raise ValueError("Index must be >= 1")
        
    row = data.iloc[i - 1]
    return {
            "tweetID": row["tweetID"],
            "tweet-text": row["tweet-text"],
            "tweet-class": row["tweet-class"]
    }

def compare (true_predict,predict_model):
    count_good_prediction = 0

    for i in range (len(predict_model)):
        if true_predict[i] == predict_model[i]:
            count_good_prediction+=1
    return count_good_prediction

def main():
    results = []

    for i in tqdm(range(1, len(df)+1)):   
        tweet_data = get_data(i, df)
        classification = classify_tweet(tweet_data["tweet-text"])
        print(f"Tweet #{i}\n (TEXT: {tweet_data['tweet-text']})\n is classified as: {classification}\n\n")
        results.append(classification) 

    #class_tweet_true = df['severity'].astype(str).tolist()
    class_tweet_true =  df['tweet-class'].tolist()
    count_good_prediction = compare (class_tweet_true,results)
    print((count_good_prediction/len(df))*100)
    

    print("\n📊 Detailed Classification Report:\n")
    report = classification_report(class_tweet_true, results, output_dict=True)
    print(classification_report(class_tweet_true, results))

    cm = confusion_matrix(class_tweet_true, results)
    labels = sorted(set(class_tweet_true))
    print("Classes uniques :", labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Réelles")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")  # Sauvegarde
    plt.close() 


    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(index=["accuracy", "macro avg", "weighted avg"])

    report_df[["precision", "recall", "f1-score"]].plot(kind="bar", figsize=(10, 6))
    plt.title("Scores par classe")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("classification_scores.png")  # Sauvegarde
    plt.close()


if __name__ == "__main__":
    main()
