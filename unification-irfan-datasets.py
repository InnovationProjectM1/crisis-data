"""
CRISIS PROJECT
Data Part
Preprocessing of the dataset from Irfan Ullah
"""
# Import libraries
import pandas as pd
import re
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

file = './Irfan-Ullah/dataset.csv'
data = pd.read_csv(file)

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

data = data.drop_duplicates()
#data['tweet-text'] = data['tweet-text'].astype(str).apply(clean_text)
example_text = "OMG 😂 I'm sooo tired rn!! Can't wait 2 sleep 😴... brb, gotta grab some coffee ☕️ #exhausted @friend http://example.com"
cleaned = clean_text(example_text)
print("\nCleaned text example:\n", example_text)
print(cleaned)

""" Note pour Giulio et Enzo pour comprendre l'exemple"""
# le "I am" de l'exemple n'apparait pas dans le final car c'est en dessous de 3 caractères 
# le @ et les url sont enlevés
# ponctuations enlevées
# emojis transformés en texte avec des _ entre les descriptions

#print("\nData after the actual clean\n", data['tweet-text'].head())
#print("\nUnique text-class values:\n", data['tweet-class'].unique())

