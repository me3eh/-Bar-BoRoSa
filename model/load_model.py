import mlflow.pytorch as mp
from transformers import BertTokenizer
import torch
import mlflow
from dotenv import load_dotenv
import os
load_dotenv("../env")


def load_model():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_SERVER"))
    logged_model = f'runs:/'+ os.environ.get('RUN_ID')+'/model'
    # with mlflow.start_run as run:
    #     mlflow.pytorch.log_model(logged_model, "model")

        # Inference after loading the logged model
    device = torch.device('cpu')

    # model = None
    # Load model as a PyFuncModel.
    model = mlflow.sklearn.load_model(logged_model)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return model, tokenizer, device


def predict(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt',
                       max_length=32, truncation=True)
    print(inputs)

    # Convert tokenized inputs to PyTorch tensors
    inputs = {key: value.to(device)
              for key, value in inputs.items() if key != 'token_type_ids'}
    print(inputs)
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits).item()

    # Map predicted label to class
    label_map = {1: 'fake', 0: 'real'}
    predicted_class = label_map[predicted_label]

    return predicted_class
