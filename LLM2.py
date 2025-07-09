import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

label_descriptions = {
    "Clothing": "requests or offers of clothes",
    "Food": "requests or offers of food or water",
    "Medical": "medical help or emergencies",
    "Money": "financial aid, donations, or funding",
    "Shelter": "requests or info about accommodation, tents, etc.",
    "Volunteer": "calls for volunteers or people offering help"
}

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def classify_tweets_batch(tweet_list, batch_size=8):
    results = []

    few_shot_examples = """
Tweet: "We need water and food urgently in the village."
Category: Food

Tweet: "Is there any shelter available near downtown?"
Category: Shelter

Tweet: "Please donate money for earthquake victims."
Category: Money

Tweet: "I can volunteer to help distribute supplies."
Category: Volunteer

Tweet: "I need clothes for my children."
Category: Clothing

Tweet: "My brother needs urgent medical attention."
Category: Medical
"""

    for i in range(0, len(tweet_list), batch_size):
        batch = tweet_list[i:i + batch_size]
        cleaned_batch = [clean_text(t) for t in batch]

        prompt = f"""
You are a tweet classifier for humanitarian crisis contexts.

Assign one of the following categories to each tweet:
{', '.join(label_descriptions.keys())}

Output only the category. Do not explain.

Examples:
{few_shot_examples.strip()}

Classify the following tweets:
"""

        for idx, tweet in enumerate(cleaned_batch, start=1):
            prompt += f"\nTweet {idx}: \"{tweet}\"\nCategory:"

        inputs = tokenizer(prompt.strip(), return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=20 * len(cleaned_batch), temperature=0.1)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extraction ligne par ligne
        lines = decoded.split("\n")
        categories = []
        for line in lines:
            for label in label_descriptions.keys():
                if label.lower() in line.lower():
                    categories.append(label)
                    break
        # Assure qu’on a bien 1 prédiction/tweet
        while len(categories) < len(cleaned_batch):
            categories.append("Unknown")

        results.extend(categories)

    return results

def get_data(i, data):
    if i < 1 or i > len(data):
        raise ValueError("Index out of range")
        
    row = data.iloc[i - 1]
    return {
        "tweetID": row.get("tweetID", ""),
        "tweet-text": row.get("tweet-text", ""),
        "tweet-class": row.get("tweet-class", "")
    }

def compare(true_labels, predicted_labels):
    count = sum(t == p for t, p in zip(true_labels, predicted_labels))
    return count

def main():
    df = pd.read_csv("./dataset/dataset_CRISIS/dataset2.csv")
    df = df.dropna(subset=["tweet-text", "tweet-class"])
    df = df.reset_index(drop=True)
    df = df.iloc[:10]

    results = []
    true_classes = []

    print(f"🔍 Starting classification on {len(df)} tweets...\n")

    for i in range(1, len(df) + 1):
        try:
             
            results = classify_tweets_batch(df["tweet-text"].tolist())
            true_classes = df["tweet-class"].tolist()

            for i, (true_label, predicted) in enumerate(zip(true_classes, results), 1):
                print(f"Tweet #{i}")
                print(f"Text     : {df['tweet-text'][i-1]}")
                print(f"Predicted: {predicted}")
                print(f"True     : {true_label}\n")


        except Exception as e:
            print(f"❌ Error on tweet #{i}: {e}")
            results.append("Unknown")
            true_classes.append("Unknown")

    correct = compare(true_classes, results)
    accuracy = (correct / len(true_classes)) * 100
    print(f"\n✅ Classification Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
