from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import pipeline
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests


def get_email(i):
    if i<1 or i>10:
        raise Exception("Invalid email number")

    # URL to download
    url = f"https://data.heatonresearch.com/wustl/CABI/genai-langchain/emails/email_{i}.txt"

    # Perform a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the content of the response to a string
        content = response.text
        return content
    else:
        raise Exception("Failed to retrieve the content")
    
model_name = "facebook/bart-large-mnli"

classifier  = pipeline("zero-shot-classification",model=model_name)

def classify_email(email):
    labels = ["spam", "faculty ", "students requesting help on an assignment", "letter of recommendation", "other"]
    result = classifier(email, candidate_labels=labels)
    return result["labels"][0]

llm = RunnableLambda(classify_email)


for i in range(1,11):
    email = get_email(i)
    classification = classify_email(email)
    if classification == 'help':
        assignment = classify_email({email})
        print(f"Email #{i} is a question about assignment #{assignment}")
    else:
        print(f"Email #{i} is: {classification}")



''' Pour une seul requete
sequence_to_classify = "Dear Professor Lawson,I m Alex Chen, leader of team Nova in the Data Science Challenge (Spring 2019).I am seeking to pursue a PhD in Computer Science, and I am hoping you could provide a recommendation letter for my applications.This term has been unexpectedly demanding, leading to delays in preparing my applications. I acknowledge the timing is not ideal, but I am committed to completing my applications diligently. Your guidance in CSC 402 Advanced Machine Learning and your support in my research project have been invaluable. I was particularly drawn to this field, prompting me to enroll in your course despite it not contributing to my major credits. The course proved to be highly beneficial, enriching my understanding through online lectures and practical assignments. I excelled in the course, securing an A+ and leading my team to the top position in the semester's Data Science Challenge. In the previous term (Fall 2019), I encountered a challenging issue in my project on Automated Segmentation of Cardiac MRI Images. After discussing this in a campus session, you introduced me to a seminal paper on convolutional networks, which greatly influenced my approach to developing an image segmentation model using advanced neural network techniques. Looking ahead, I aim to delve deeper into Machine Learning and computational models. The knowledge acquired from your course is sure to be a significant advantage in my future studies. I have attached my most recent CV and academic transcript for your review. I am applying to approximately 10 universities, with the earliest deadline on January 5th. I would appreciate the opportunity to discuss this further. Thank you very much, and I hope you have a wonderful holiday season!Best regards,Alex Chen"
candidate_labels = ["spam", "faculty", "help", "letter of recommendation", "other"]
print(classifier(sequence_to_classify, candidate_labels))'''