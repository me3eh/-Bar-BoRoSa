import pickle
import torch
from transformers import BertTokenizer
import io
import pandas as pd
import re

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='mps')
        else: return super().find_class(module, name)



device = torch.device('mps')
with open('./second_iteration.pkl', 'rb') as f:
    loaded_model = CPU_Unpickler(f).load() 


loaded_model.to(device)

model = loaded_model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

label_map = {1: 'fake', 0: 'real'}


df = pd.read_csv('WELFake_Dataset.csv')

df['label'] = df['label'].apply(lambda x: 'fake' if x == 0 else 'real')
def preprocess(text):
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip().lower()
df.dropna(inplace=True)


print(df.head())

df['text'] = df['text'].apply(preprocess)
df = df[['text', 'label']]

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and preprocess the data
tokenized_texts = []
for text in df['text']:
    inputs = tokenizer(text, return_tensors='pt', max_length=32, truncation=True)
    tokenized_texts.append(inputs)

# Convert tokenized inputs to PyTorch tensors
input_tensors = []
for inputs in tokenized_texts:
    inputs.to(device)  # Assuming you have defined the device earlier
    input_tensors.append(inputs)

# Run inference
predictions = []
with torch.no_grad():
    for inputs in input_tensors:
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits).item()
        predicted_class = label_map[predicted_label]
        predictions.append(predicted_class)

# Add predictions to DataFrame
df['predicted_class'] = predictions

# Save or print the DataFrame
print(df.head())
df.to_csv('predictions_test3.csv', index=False)