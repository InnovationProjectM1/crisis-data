"""
CRISIS PROJECT
Data Part
Preprocessing of the dataset from Irfan Ullah
"""

# Import libraries
# from nltk.corpus import stopwords
# import pandas as pd
# import re
# from nltk.stem import WordNetLemmatizer

import requests
import json

base_url = "https://api.crisis.maxlamenace.duckdns.org/"

def get_tweets():
    """
    Fetch tweets from the Crisis API.
    """
    cpt = 0
    response = requests.get(base_url + "tweets")
    if response.status_code == 200:
        for tweet in response.json():
            print(tweet['classifier'])
            print('\n-----------------------\n')
            cpt += 1
        print(f"Total tweets fetched: {cpt}")
        return response.json()
    else:
        print("Error fetching tweets:", response.status_code)
        return []

tweet_data = [
        {
            "tweet_id": "264700000000000000",
            "classified_group": "Ressource",
            "classified_sub_group": "Shelter",
            "difficulty": "1"
        },
        {
            "tweet_id": "262897000000000000",
            "classified_group": "Needs",
            "classified_sub_group": "Humanitarian Aid",
            "difficulty": "5"
        }
    ]

def post_multiple_classifiers(classified_tweets):
    """
    Post classified tweets to the Crisis API.
    """

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
    

#get_tweets()
print(post_multiple_classifiers(tweet_data))

# # Initialize NLTK resources
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# # Path to the dataset
# file = './Irfan-Ullah/dataset.csv'

# # Load the data
# data = pd.read_csv(file)
# print("\n\nData from the csv\n", data['tweet-text'].head())

# def clean_text(text):
#     # Remove URLs, usernames, non-ascii characters, punctuation, stopwords, word that get a lenght <2 and convert to lowercase, lemmatize
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text)
#     text = re.sub(r'@\w+', '', text) # For usernames
#     text = text.encode('ascii', 'ignore').decode('ascii') 
#     text = text.lower() 
#     text = re.sub(r'[^\w\s]', '', text) # For punctuation
#     words = [word for word in text.split() if word not in stop_words and len(word) > 2]
#     lemmatized = [lemmatizer.lemmatize(word) for word in words]
#     final = [word.strip() for word in lemmatized if word.strip()]      
#     return ' '.join(final)

# # Start of the data preprocessing
# data = data.drop_duplicates()
# data['tweet-text'] = data['tweet-text'].apply(clean_text)
# print("\nData after the actual clean\n", data['tweet-text'].head())

# # Print unique text-class values
# print("\nUnique text-class values:\n", data['tweet-class'].unique())
