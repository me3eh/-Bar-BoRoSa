import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pytorch_lightning as pl
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv
import os

load_dotenv("../../../.env")

mlflow.set_tracking_uri(os.environ.get("MLFLOW_SERVER"))

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

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()

        attention_mask = torch.ones_like(input_ids)

        if 'attention_mask' in encoding:
            attention_mask = encoding['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }


class BertClassifier(pl.LightningModule):
    def __init__(self, num_labels, learning_rate=2e-5, dropout_prob=0.1):
        super(BertClassifier, self).__init__()
        self.save_hyperparameters()  # This line logs all hyperparameters

        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=num_labels)
        self.learning_rate = learning_rate
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        logits = outputs.logits

        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss)  # Log training loss

        # Example: Log additional metrics
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        # Log training accuracy
        self.log('train_accuracy', accuracy, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        logits = outputs.logits

        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss)  # Log validation loss

        # Example: Log additional metrics
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        # Log validation accuracy
        self.log('val_accuracy', accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        logits = outputs.logits

        loss = F.cross_entropy(logits, labels)
        self.log('test_loss', loss)  # Log test loss

        # Example: Log additional metrics
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        self.log('test_accuracy', accuracy)  # Log test accuracy

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate)
        return optimizer


def preprocess(text):
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip().lower()


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data = pd.read_csv('./model/data/WELFake_Dataset.csv')
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)

    data['text'] = data['text'] + ' ' + data['title']
    data['text'] = data['text'].apply(preprocess)

    x_train, x_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.3, random_state=42)

    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    train_dataset = FakeNewsDataset(x_train, y_train, tokenizer, max_length=32)
    eval_dataset = FakeNewsDataset(x_test, y_test, tokenizer, max_length=32)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16)

    model = BertClassifier(num_labels=2)

    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss")

    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[early_stopping],
        accelerator='mps' if torch.cuda.is_available() else 'mps',
    )
    mlflow.pytorch.autolog()
    # Initiate MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(model.hparams)

        input_example = {
            "text": "asdasda",
        }
        # Log model architecture
        signature = infer_signature({"input": "string"}, {"output": "int"})

        mlflow.pytorch.log_model(
            model, "models", input_example=input_example, signature=signature)

        # Train the model
        trainer.fit(model, train_loader, eval_loader)

        # Evaluate on test set
        trainer.test(model, eval_loader)


if __name__ == "__main__":
    main()
