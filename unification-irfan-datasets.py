"""
CRISIS PROJECT
This script combines two datasets and clean the text data from Irfan Ullah into a single CSV file.
"""
# Import libraries
from nltk.corpus import stopwords
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Path to the dataset
file = './Irfan-Ullah/Irfan-Ullah/dataset2.csv'

# Load the data
data = pd.read_csv(file)
print("Data from the csv", data['tweet-text'].head())

def clean_text(text):
    # Remove URLs, usernames, non-ascii characters, punctuation, stopwords, word that get a lenght <2 and convert to lowercase, lemmatize
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text) # For usernames
    text = text.encode('ascii', 'ignore').decode('ascii') 
    text = text.lower() 
    text = re.sub(r'[^\w\s]', '', text) # For punctuation
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    final = [word.strip() for word in lemmatized if word.strip()]      
    return ' '.join(final)

data = data.drop_duplicates()
data['tweet-text'] = data['tweet-text'].apply(clean_text)
print("Data after the actual clean", data['tweet-text'].head())

# Print unique text-class values
print("Unique text-class values:", data['tweet-class'].unique())
