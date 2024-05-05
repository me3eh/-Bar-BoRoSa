import mlflow.pytorch
from transformers import BertTokenizer
import torch

from dotenv import load_dotenv
import os
load_dotenv(".env")
def load_model():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_SERVER"))   
    logged_model = 'runs:/3a33c32202da47999ac3c474edad9618/model'

    device = torch.device('cpu')

    # Load model as a PyFuncModel.
    model = mlflow.pytorch.load_model(logged_model)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return model, tokenizer, device


def predict(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt',
                       max_length=32, truncation=True)

    # Convert tokenized inputs to PyTorch tensors
    inputs = {key: value.to(device)
              for key, value in inputs.items() if key != 'token_type_ids'}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits).item()

    # Map predicted label to class
    label_map = {1: 'fake', 0: 'real'}
    predicted_class = label_map[predicted_label]

    return predicted_class
