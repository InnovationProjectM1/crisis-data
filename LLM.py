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
    

#print(get_email(1))


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import pipeline
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
import torch


model_name = "meta-llama/Meta-Llama-3-70B"  #meta-llama/Meta-Llama-3-8B

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",    # utilise les GPUs disponibles automatiquement
    torch_dtype=torch.float16,  # 16 bits pour économiser mémoire
)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.5,
)

llm = RunnableLambda(lambda prompt: hf_pipeline(prompt)[0]['generated_text'].strip())

email_prompt = PromptTemplate( input_variables = ['email'], template = """
You are an email classifier. Read the email and respond with one word: spam, faculty, help, lor, or other.
Categories:

spam: marketing/promotional emails
faculty: official communications from staff/administration
help: students asking for academic assistance
lor: requests for recommendation letters
other: anything else

Email: {email}
Classification:""")

help_prompt = PromptTemplate( input_variables = ['email'], template = """
You are given an email where a student is asking about an assignment. Return the assignment number
that they are asking about. If you cannot tell return a ?. Return only the assignment number as
an integer, do not explain.
Here is the email:

{email}""")

chain_email = email_prompt  | llm
chain_help = help_prompt  |  llm

for i in range(1,11):
    email = get_email(i)
    classification_prompt = email_prompt.format_prompt(email=email).to_string()
    classification = llm.invoke(classification_prompt).strip().lower()
    print("\n\n\nClassification\n\n\n",classification)
    if classification == 'help':
        help_prompt_text = help_prompt.format_prompt(email=email).to_string()
        assignment = llm.invoke(help_prompt_text).strip()
        print(f"Email #{i} is a question about assignment #{assignment}")
    else:
        print(f"Email #{i} is: {classification}")

