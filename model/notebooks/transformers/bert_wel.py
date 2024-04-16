# importing required libaries
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torchinfo import summary
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import re

import mlflow

mlflow.set_tracking_uri(uri="http://localhost:8080")


model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2)


class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # get the input ids
        input_ids = encoding['input_ids'].flatten()

        # create the attention mask
        attention_mask = torch.ones_like(input_ids)

        # if 'attention_mask' exists in encoding, use it
        if 'attention_mask' in encoding:
            attention_mask = encoding['attention_mask'].flatten()
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': torch.tensor(label, dtype=torch.long)
            }


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)

# Example function for evaluating the model


def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(eval_loader), accuracy


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocess(text):
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip().lower()


#  First Dataset
data = pd.read_csv('./WELFake_Dataset.csv')
data = data[:10000]
# handle duplicated values
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)  # Remove rows with missing values


data['text'] = data['text'] + ' ' + data['title']
data['text'] = data['text'].apply(preprocess)


BATCH_SIZE = 16
MAX_LENGTH = 32
LEARNING_RATE = 2e-5
EPOCHS = 2

x_train, x_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.3, random_state=42)

x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


train_dataset = FakeNewsDataset(x_train, y_train, tokenizer, MAX_LENGTH)
eval_dataset = FakeNewsDataset(x_test, y_test, tokenizer, MAX_LENGTH)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

device = torch.device('mps')
model.to(device)

with mlflow.start_run():
    params = {
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "loss_function": criterion.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
    }
    # Log training parameters.
    mlflow.log_params(params)

    # Log model summary.
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        eval_loss, eval_accuracy = evaluate(
            model, eval_loader, criterion, device)
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(
            f'Evaluation Loss: {eval_loss:.4f} | Evaluation Accuracy: {eval_accuracy:.4f}')

    # Save the trained model to MLflow.
    mlflow.pytorch.log_model(model, "model")
