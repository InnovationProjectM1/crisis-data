from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import pipeline
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
import contractions
import emoji
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_data(i, data):
    if i < 1:
        raise ValueError("Index must be >= 1")
        
    row = data.iloc[i - 1]
    return {
            "tweetID": row["tweetID"],
            "tweet-text": row["tweet-text"],
            "tweet-class": row["tweet-class"]
    }

class LLM:
    
    def __init__(self):
        self.model_name = "facebook/bart-large-mnli"

        self.classifier  = pipeline("zero-shot-classification",model=self.model_name)

    def classify_tweet(self,tweet_text):
        labels = {
            "Clothing": "requests or offers of clothes",
            "Food": "requests or offers of food or water",
            "Medical": "medical help or emergencies",
            "Money": "financial aid, donations, or funding",
            "Shelter": "requests or info about accommodation, tents, etc.",
            "Volunteer": "calls for volunteers or people offering help"
        }
        reverse_labels = {v: k for k, v in labels.items()}
        result = self.classifier(tweet_text, candidate_labels=list(labels.values()))
        predicted_description = result["labels"][0]
        predicted_label = reverse_labels[predicted_description]
        return predicted_label

def preprocess(file):
    
    abbreviations = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "y": "why",
    "b": "be",
    "c": "see",
    "k": "okay",
    "ok": "okay",
    "thx": "thanks",
    "ty": "thank you",
    "yw": "you are welcome",
    "plz": "please",
    "pls": "please",
    "msg": "message",
    "txt": "text",
    "tmrw": "tomorrow",
    "btw": "by the way",
    "idk": "i do not know",
    "idc": "i do not care",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "omg": "oh my god",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing",
    "tbh": "to be honest",
    "fyi": "for your information",
    "brb": "be right back",
    "bbl": "be back later",
    "gtg": "got to go",
    "g2g": "got to go",
    "bfn": "bye for now",
    "np": "no problem",
    "wtf": "what the fuck",
    "wth": "what the hell",
    "afaik": "as far as i know",
    "asap": "as soon as possible",
    "irl": "in real life",
    "jk": "just kidding",
    "nvm": "never mind",
    "ily": "i love you",
    "ilu": "i love you",
    "ttyl": "talk to you later",
    "smh": "shaking my head",
    "ikr": "i know right",
    "dm": "direct message",
    "ama": "ask me anything",
    "ftw": "for the win",
    "hbu": "how about you",
    "hbd": "happy birthday",
    "gr8": "great",
    "bday": "birthday",
    "sec": "second",
    "pics": "pictures",
    "pic": "picture",
    "bc": "because",
    "cya": "see you",
    "tbf": "to be fair",
    "omw": "on my way",
    "sry": "sorry",
    "u2": "you too"
    }

    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()


    # Load the data
    data = pd.read_csv(file)
    
    def replace_abbreviations(text, abbr_dict):
        pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in abbr_dict.keys()) + r')\b')
        return pattern.sub(lambda x: abbr_dict[x.group().lower()], text)

    def clean_text(text):
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = contractions.fix(text)

        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text) # For usernames
        text = text.encode('ascii', 'ignore').decode('ascii') 
        text = text.lower() 
        text = re.sub(r'[^\w\s]', '', text) # For punctuation

        text = replace_abbreviations(text, abbreviations)

        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        final = [word.strip() for word in lemmatized if word.strip()]      
        return ' '.join(final)

    # Start of the data preprocessing
    data = data.drop_duplicates()
    data['tweet-text'] = data['tweet-text'].apply(clean_text)
    return data

def compare (true_predict,predict_model):
    count_good_prediction = 0

    for i in range (len(predict_model)):
        if true_predict[i] == predict_model[i]:
            count_good_prediction+=1
    return count_good_prediction

def main():
    file = './dataset/dataset_CRISIS/dataset2.csv'
    data = preprocess(file)
    llm = LLM()
    results = []

    for i in range(1, len(data)+1):   
        tweet_data = get_data(i, data)  # NE PAS ÉCRASER `data`
        classification = llm.classify_tweet(tweet_data["tweet-text"])  # Passer le texte uniquement
        print(f"Tweet #{i}\n (TEXT: {tweet_data['tweet-text']})\n is classified as: {classification}\n\n")
        results.append(classification) 

    class_tweet_true =  data['tweet-class'].tolist()
    count_good_prediction = compare (class_tweet_true,results)
    print((count_good_prediction/len(data))*100)

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
    plt.savefig("confusion_matrix_bart_large_mnlis.png")  # Sauvegarde
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
    plt.savefig("classification_score_bart_large_mnlis.png")  # Sauvegarde
    plt.close()



if __name__ == "__main__":
    main()