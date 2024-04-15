{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XTOKeeBnI-V0"
      },
      "source": [
        "# Fake News Detection With Transformer Models XLNET"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 48,
      "metadata": {
        "id": "bPtajH4yI-V2"
      },
      "outputs": [],
      "source": [
        "# importing required libaries\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.metrics import classification_report\n",
        "from sklearn.metrics import confusion_matrix\n",
        "import pickle\n",
        "from transformers import XLNetConfig, XLNetModel\n",
        "import torch\n",
        "from transformers import XLNetTokenizer, XLNetForSequenceClassification\n",
        "from transformers import AdamW\n",
        "from torch.utils.data import DataLoader, Dataset\n",
        "import re\n",
        "import pickle"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "from google.colab import drive\n",
        "MOUNTPOINT = '/content/gdrive'\n",
        "DATADIR = os.path.join(MOUNTPOINT, 'My Drive', 'myfolder')\n",
        "drive.mount(MOUNTPOINT)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3WMD6YsGjQBs",
        "outputId": "7ccd0bdd-497c-4c6c-c0ea-9fdad28817f3"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount(\"/content/gdrive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "YWtUhns5jd8f"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 49,
      "metadata": {
        "id": "DPhigfd6I-V3"
      },
      "outputs": [],
      "source": [
        "\n",
        "# Initializing a XLNet configuration\n",
        "configuration = XLNetConfig()\n",
        "\n",
        "# Initializing a model (with random weights) from the configuration\n",
        "model = XLNetModel(configuration)\n",
        "\n",
        "# Accessing the model configuration\n",
        "configuration = model.config"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 50,
      "metadata": {
        "id": "LU8LGCOYI-V4"
      },
      "outputs": [],
      "source": [
        "class FakeNewsDataset(Dataset):\n",
        "    def __init__(self, texts, labels, tokenizer, max_length):\n",
        "        self.texts = texts\n",
        "        self.labels = labels\n",
        "        self.tokenizer = tokenizer\n",
        "        self.max_length = max_length\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.texts)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "       text = str(self.texts[idx])\n",
        "       label = self.labels[idx]\n",
        "\n",
        "       # tokenize the text\n",
        "       encoding = self.tokenizer.encode_plus(\n",
        "           text,\n",
        "           add_special_tokens=True,\n",
        "           max_length=self.max_length,\n",
        "           return_token_type_ids=False,\n",
        "           padding='max_length',\n",
        "           truncation=True,\n",
        "           return_tensors='pt'\n",
        "       )\n",
        "\n",
        "       # get the input ids\n",
        "       input_ids = encoding['input_ids'].flatten()\n",
        "\n",
        "       # create the attention mask\n",
        "       attention_mask = torch.ones_like(input_ids)\n",
        "\n",
        "       # if 'attention_mask' exists in encoding, use it\n",
        "       if 'attention_mask' in encoding:\n",
        "           attention_mask = encoding['attention_mask'].flatten()\n",
        "           return {\n",
        "           'input_ids': input_ids,\n",
        "           'attention_mask': attention_mask,\n",
        "           'label': torch.tensor(label, dtype=torch.long)\n",
        "       }"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 33,
      "metadata": {
        "id": "V05gIxy_I-V4"
      },
      "outputs": [],
      "source": [
        "def train(model, train_loader, optimizer, criterion, device):\n",
        "    model.train()\n",
        "    total_loss = 0.0\n",
        "    for batch in train_loader:\n",
        "        input_ids = batch['input_ids'].to(device)\n",
        "        attention_mask = batch['attention_mask'].to(device)\n",
        "        labels = batch['label'].to(device)\n",
        "\n",
        "        optimizer.zero_grad()\n",
        "        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)\n",
        "        loss = outputs.loss\n",
        "        total_loss += loss.item()\n",
        "\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "\n",
        "    return total_loss / len(train_loader)\n",
        "\n",
        "# Example function for evaluating the model\n",
        "def evaluate(model, eval_loader, criterion, device):\n",
        "    model.eval()\n",
        "    total_loss = 0.0\n",
        "    correct = 0\n",
        "    total = 0\n",
        "    with torch.no_grad():\n",
        "        for batch in eval_loader:\n",
        "            input_ids = batch['input_ids'].to(device)\n",
        "            attention_mask = batch['attention_mask'].to(device)\n",
        "            labels = batch['label'].to(device)\n",
        "\n",
        "            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)\n",
        "            loss = outputs.loss\n",
        "            total_loss += loss.item()\n",
        "\n",
        "            predictions = torch.argmax(outputs.logits, dim=1)\n",
        "            correct += (predictions == labels).sum().item()\n",
        "            total += labels.size(0)\n",
        "\n",
        "    accuracy = correct / total\n",
        "    return total_loss / len(eval_loader), accuracy\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 51,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AuSeICdkI-V5",
        "outputId": "d88eec34-6a93-43fa-be92-ce8bfb1d6467"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.bias', 'logits_proj.weight', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
          ]
        }
      ],
      "source": [
        "tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')\n",
        "model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)\n",
        "from sklearn.model_selection import train_test_split"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 35,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "P5345SelVr8e",
        "outputId": "c2eaa4cf-0c8e-4735-91c4-4512b19f88a7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "mkdir: cannot create directory ‘/root/.kaggle’: File exists\n",
            "Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /root/.kaggle/kaggle.json'\n",
            "fake-news-detection.zip: Skipping, found more recently modified local copy (use --force to force download)\n"
          ]
        }
      ],
      "source": [
        "! pip install -q kaggle\n",
        "! mkdir ~/.kaggle\n",
        "! touch ~/.kaggle/kaggle.json\n",
        "!echo '{\"username\":\"akselsaatci\",\"key\":\"asdasdasd\"}' > ~/.kaggle/kaggle.json\n",
        "! kaggle datasets download -d vishakhdapat/fake-news-detection"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 36,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KRuKgNR1YBqi",
        "outputId": "e4c5b3a6-8a51-4927-fdcb-7e5267730195"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Archive:  fake-news-detection.zip\n",
            "replace fake_and_real_news.csv? [y]es, [n]o, [A]ll, [N]one, [r]ename: "
          ]
        }
      ],
      "source": [
        "! unzip fake-news-detection.zip"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 52,
      "metadata": {
        "id": "pzjnvrHZJzJK"
      },
      "outputs": [],
      "source": [
        "data = pd.read_csv('./fake_and_real_news.csv')\n",
        "# handle duplicated values\n",
        "data.drop_duplicates(inplace=True)\n",
        "data.dropna(inplace=True)  # Remove rows with missing values\n",
        "\n",
        "\n",
        "data['is_fake'] = 0\n",
        "\n",
        "data['is_fake'] = (data['label'] == 'Fake').astype(int)\n",
        "\n",
        "def preprocess(text):\n",
        "    text = re.sub(r'[^\\w\\s\\']', ' ', text)\n",
        "    text = re.sub(r' +', ' ', text)\n",
        "    return text.strip().lower()\n",
        "\n",
        "data['Text'] = data['Text'].apply(preprocess)\n",
        "\n",
        "\n",
        "x_train, x_test, y_train, y_test = train_test_split(data['Text'], data['is_fake'], test_size=0.3, random_state=0)\n",
        "\n",
        "x_train.reset_index(drop=True, inplace=True)\n",
        "x_test.reset_index(drop=True, inplace=True)\n",
        "y_train.reset_index(drop=True, inplace=True)\n",
        "y_test.reset_index(drop=True, inplace=True)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "filtered_y_test = [y_test[i] for i in range(len(y_test)) if y_test[i] == 1]\n",
        "print(filtered_y_test)\n",
        "\n",
        "filtered_y_train = [y_train[i] for i in range(len(y_train)) if y_train[i] == 1]\n",
        "print(filtered_y_train)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6JvUHRqwxDy6",
        "outputId": "e20d8678-6337-4888-f141-5aed8e36632b"
      },
      "execution_count": 55,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]\n",
            "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 61,
      "metadata": {
        "id": "OaUJvqXfJ5Gz"
      },
      "outputs": [],
      "source": [
        "BATCH_SIZE = 8\n",
        "MAX_LENGTH = 32\n",
        "LEARNING_RATE = 2e-5\n",
        "EPOCHS = 8\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 62,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "S7H8ts-fKNMG",
        "outputId": "49966b7b-5560-4457-e9a7-91b8ff9ab70f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/8:\n",
            "Training Loss: 0.0732\n",
            "Evaluation Loss: 0.0122 | Evaluation Accuracy: 0.9976\n",
            "Epoch 2/8:\n",
            "Training Loss: 0.0033\n",
            "Evaluation Loss: 0.0110 | Evaluation Accuracy: 0.9983\n",
            "Epoch 3/8:\n",
            "Training Loss: 0.0000\n",
            "Evaluation Loss: 0.0119 | Evaluation Accuracy: 0.9990\n",
            "Epoch 4/8:\n",
            "Training Loss: 0.0136\n",
            "Evaluation Loss: 0.0112 | Evaluation Accuracy: 0.9983\n",
            "Epoch 5/8:\n",
            "Training Loss: 0.0026\n",
            "Evaluation Loss: 0.0103 | Evaluation Accuracy: 0.9990\n",
            "Epoch 6/8:\n",
            "Training Loss: 0.0009\n",
            "Evaluation Loss: 0.0114 | Evaluation Accuracy: 0.9990\n",
            "Epoch 7/8:\n",
            "Training Loss: 0.0000\n",
            "Evaluation Loss: 0.0120 | Evaluation Accuracy: 0.9990\n",
            "Epoch 8/8:\n",
            "Training Loss: 0.0000\n",
            "Evaluation Loss: 0.0125 | Evaluation Accuracy: 0.9990\n"
          ]
        }
      ],
      "source": [
        "train_dataset = FakeNewsDataset(x_train, y_train, tokenizer, MAX_LENGTH)\n",
        "eval_dataset = FakeNewsDataset(x_test, y_test, tokenizer, MAX_LENGTH)\n",
        "\n",
        "    # Create data loaders\n",
        "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)\n",
        "eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)\n",
        "\n",
        "    # Move model to GPU if available\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "model.to(device)\n",
        "\n",
        "    # Define loss function and optimizer\n",
        "criterion = torch.nn.CrossEntropyLoss()\n",
        "optimizer =  torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)\n",
        "\n",
        "    # Training loop\n",
        "for epoch in range(EPOCHS):\n",
        "  train_loss = train(model, train_loader, optimizer, criterion, device)\n",
        "  eval_loss, eval_accuracy = evaluate(model, eval_loader, criterion, device)\n",
        "  print(f'Epoch {epoch+1}/{EPOCHS}:')\n",
        "  print(f'Training Loss: {train_loss:.4f}')\n",
        "  print(f'Evaluation Loss: {eval_loss:.4f} | Evaluation Accuracy: {eval_accuracy:.4f}')\n",
        "\n",
        "\n",
        "with open(\"./gdrive/MyDrive/changed_with_eight_epocs_xlnet.pkl\", \"wb\") as file: # file is a variable for storing the newly created file, it can be anything.\n",
        "    pickle.dump(model, file) # Dump function is used to write the object into the created file in byte format.\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "ZI4eBSNRso30"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 77,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dCcdxl3pLXx5",
        "outputId": "68bc42a6-721c-481e-9b72-68ad55f2868d"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "XLNetForSequenceClassification(\n",
              "  (transformer): XLNetModel(\n",
              "    (word_embedding): Embedding(32000, 768)\n",
              "    (layer): ModuleList(\n",
              "      (0-11): 12 x XLNetLayer(\n",
              "        (rel_attn): XLNetRelativeAttention(\n",
              "          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "          (dropout): Dropout(p=0.1, inplace=False)\n",
              "        )\n",
              "        (ff): XLNetFeedForward(\n",
              "          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "          (layer_1): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          (layer_2): Linear(in_features=3072, out_features=768, bias=True)\n",
              "          (dropout): Dropout(p=0.1, inplace=False)\n",
              "          (activation_function): GELUActivation()\n",
              "        )\n",
              "        (dropout): Dropout(p=0.1, inplace=False)\n",
              "      )\n",
              "    )\n",
              "    (dropout): Dropout(p=0.1, inplace=False)\n",
              "  )\n",
              "  (sequence_summary): SequenceSummary(\n",
              "    (summary): Linear(in_features=768, out_features=768, bias=True)\n",
              "    (activation): Tanh()\n",
              "    (first_dropout): Identity()\n",
              "    (last_dropout): Dropout(p=0.1, inplace=False)\n",
              "  )\n",
              "  (logits_proj): Linear(in_features=768, out_features=2, bias=True)\n",
              ")"
            ]
          },
          "metadata": {},
          "execution_count": 77
        }
      ],
      "source": [
        "with open('./gdrive/MyDrive/changed_with_eight_epocs_xlnet.pkl', 'rb') as f:\n",
        "    loaded_model = pickle.load(f)\n",
        "\n",
        "# Ensure to move the model to the appropriate device if necessary\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "loaded_model.to(device)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 78,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4UnwwLWXK0pE",
        "outputId": "7acbd3ab-a914-4b28-c9c5-548deb2dee02"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Predicted class: fake\n"
          ]
        }
      ],
      "source": [
        "\n",
        "# Load your XLNet model and tokenizer\n",
        "model = loaded_model\n",
        "\n",
        "input_text = \"Anthem to pare back Obamacare offerings in Nevada and Georgia (Reuters) - U.S. health insurer Anthem Inc (ANTM.N) said on Monday it will no longer offer Obamacare plans in Nevada’s state exchange and will stop offering the plans in nearly half of Georgia’s counties next year. The moves come after Republican senators last month failed to repeal and replace Obamacare, former President Barack Obama’s signature healthcare reform law, creating uncertainty over how the program providing health benefits to 20 million Americans will be funded and managed in 2018. Hundreds of U.S. counties are at risk of losing access to private health coverage in 2018 as insurers consider pulling out of those markets in the coming months. Nevada had said in June that residents in 14 counties out of 17 in the state would not have access to qualified health plans on the state exchanges. Anthem’s decision to leave the state entirely does not increase the number of “bare counties” in the state, Nevada Insurance Commissioner Barbara Richardson said in a statement. The insurer will still offer “catastrophic plans,” which can be purchased outside the state’s exchange and are only available to consumers under 30 years old or with a low income. Anthem also said it will only offer Obamacare plans in 85 of Georgia’s 159 counties. It said the counties it will continue to offer the plans in are mostly rural counties that would otherwise not have health insurance coverage for their residents.  It said these changes do not impact Anthem’s Medicare Advantage, Medicaid or employer-based plans in either state.   The company said last week that it will pull out of 16 of 19 pricing regions in California in 2018 where it offered Obamacare options this year.  Anthem blamed the moves in part on uncertainty over whether the Trump administration would maintain subsidies that keep costs down.  U.S. President Donald Trump last week threatened to cut off subsidy payments that make the plans affordable for lower-income Americans and help insurers to keep premiums down, after efforts to repeal the law signed by his predecessor, President Barack Obama, failed in Congress.  Trump has repeatedly urged Republican lawmakers to keep working to undo Obama’s Affordable Care Act.\"\n",
        "\n",
        "inputs = tokenizer(input_text, return_tensors='pt', max_length=32 ,truncation=True)\n",
        "\n",
        "inputs.to(device)\n",
        "\n",
        "with torch.no_grad():\n",
        "    outputs = model(**inputs)\n",
        "\n",
        "predicted_label = torch.argmax(outputs.logits).item()\n",
        "\n",
        "label_map = {1: 'fake', 0: 'real'}\n",
        "predicted_class = label_map[predicted_label]\n",
        "\n",
        "print(\"Predicted class:\", predicted_class)"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "KgmiSSGytsmI"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.8"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}