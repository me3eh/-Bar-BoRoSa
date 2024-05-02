import pickle
import torch
from transformers import BertTokenizer
import io
import pandas as pd

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



input_text = input("Enter the text to classify: ")

inputs = tokenizer(input_text, return_tensors='pt', max_length=32 ,truncation=True)

inputs.to(device)

with torch.no_grad():
    outputs = model(**inputs)

predicted_label = torch.argmax(outputs.logits).item()

predicted_class = label_map[predicted_label]

print("Predicted class:", predicted_class)