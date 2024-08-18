import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DebertaV2Tokenizer, DebertaForSequenceClassification, Trainer, TrainingArguments
import torch

# # Load your dataset
# df = pd.read_csv('stock_data.csv')
# df = df.rename(columns={'Text': 'text', 'Sentiment': 'label'})
# df['label'] = df['label'].replace({-1: 0, 1: 1})
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=10)
# train_dataset = Dataset.from_pandas(train_df)
# test_dataset = Dataset.from_pandas(test_df)

# # Load tokenizer and model
# tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
# model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2, ignore_mismatched_sizes=True)

# # Tokenize the datasets
# def tokenize_function(examples):
#     return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
# tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy='epoch',
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_test_dataset,
# )

# # Train the model
# trainer.train()
# eval_results = trainer.evaluate()

# # Save the tokenizer and model
# model.save_pretrained('./deberta_model')
# tokenizer.save_pretrained('./deberta_tokenizer')

# Load the tokenizer and model
tokenizer = DebertaV2Tokenizer.from_pretrained('./deberta_tokenizer')
model = DebertaForSequenceClassification.from_pretrained('./deberta_model')

# Define a function for prediction
def predict(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1).item()
    return 'positive' if predictions == 1 else 'negative'

# Example usage
text = "Chinese Imports Are Rising Again. Here’s What It Means for U.S. Jobs"
sentiment = predict(text)
print(f"Sentiment: {sentiment}")

from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "AdaptLLM/finance-LLM"
advice_model = AutoModelForCausalLM.from_pretrained(model_name)
advice_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Combine into a single input text
input_text = f"News: {text} The news has sentiment: {sentiment}. Relevant stock advice based on news and sentiment:"

# Tokenize and generate text
inputs = advice_tokenizer(input_text, return_tensors="pt")
outputs = advice_model.generate(inputs['input_ids'], max_length=150)

# Decode and print the generated advice
generated_advice = advice_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_advice)
