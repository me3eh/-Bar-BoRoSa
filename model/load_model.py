import mlflow.pytorch as mp
from transformers import BertTokenizer
import torch
import mlflow
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import sys
from keras.preprocessing.text import Tokenizer
import re
from keras.preprocessing.sequence import pad_sequences

# sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/notebooks")
    # import os
vectorizer = pickle.load(open(os.path.dirname(os.path.realpath(__file__))+"/data/vectorizer.pickle", 'rb')) #Load vectorizer
def preprocess(text):
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip().lower()

def load_model():
    original_file_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(f"{original_file_path}/notebooks")

    model = mlflow.sklearn.load_model('Decision Tree')
    os.chdir(original_file_path)

    return model


# def predict(text, model, tokenizer, device):
def predict(text, model):
    global vectorizer
    text = preprocess(text)
    word_vector = vectorizer.transform([text])
    prediction = model.predict(word_vector)
    final_result = "true" if prediction == 1 else "false" 
    print("The prediction for the word '{}' is: {}".format(text, prediction))
    return final_result


def load_model_nn():
    original_file_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(f"{original_file_path}/notebooks")

    model = mlflow.sklearn.load_model('Neural network')
    os.chdir(original_file_path)

    return model


def predict_nn(text, model):
    tokenizer = Tokenizer(num_words=5000, lower=True)
    text = _preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    # print(model.predict(padded_sequence))
    prediction = (model.predict(padded_sequence) < 0.5).astype("int32")
    return "false" if prediction[0][0] == 1 else "true"
# def check_fake_news(text):
    

def _preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation
    text = text.lower()  # convert to lowercase
    return text

# text = "Court Forces Ohio To Allow Millions Of Illegally Purged Voters Back On The Rolls Donald Trump is semi-right: this election might be rigged. However, this election is rigged against the people, not him.Take a look at Ohio, a key battleground state in the never-ending 2016 election. After purging more than two million voters from the roles, a high court smacked down Republican Secretary of State John Hustad for violating the National Voter Registration Act.Hustad s purge, which included some dead people and those who moved out of state, also included those who moved in the same county and those who have not voted in past elections (at least since 2011). Those who were purged were overwhelmingly black, low-income and Democratic voters. A Reuters investigation found that in the state s major cities (Cleveland, Columbus, and Cincinnati) the voters in Democratic-leaning neighborhoods and precincts were illegally purged at twice the rate as in Republican.So, on Wednesday night, while everyone was watching the final presidential debate (in which Trump again claimed the election is rigged against him), the United States District Court for the Southern District of Ohio, at the behest of Republican appointee Judge George C. Smith, ordered the voters save the dead and moved back onto the rolls immediately and have their voting rights restored.Judge Smith accused Hustad of voter disenfranchisement, writing:If those who were unlawfully removed from the voter rolls are not allowed to vote, then the Secretary of State is continuing to to disenfranchise voters in violation of federal law.The case was originally heard by the 6th U.S. Circuit Court of Appeals, which also ruled against Hustad, but was sent back to the District Court for a rehearing. Obviously, both courts realized the illegal power grab by the state at the behest of the GOP, and they weren t having it.Time and time again the courts have struck down illegal rigging practices by the Republicans, thus saving the nation from an unbalanced electoral system. Voting is a right in this country, and partisan politicos should not have the power to take it away just because.If anyone can complain about a rigged election, it s Hillary Clinton.Featured image via John Sommers II/Getty Images"
# model = load_model_nn()
# print(predict_nn(text, model))