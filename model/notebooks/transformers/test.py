import pickle
import torch
from transformers import BertTokenizer
import io
import pandas as pd


from dotenv import load_dotenv
import os

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='mps')
        else: return super().find_class(module, name)


load_dotenv("../../../.env")

device = torch.device('mps')
import mlflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_SERVER"))
logged_model = 'runs:/d808fa285c3e40e6a436c48c6fa90f96/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pytorch.load_model(logged_model)



loaded_model.to(device)

model = loaded_model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

label_map = {0: 'fake', 1: 'real'}


df = pd.read_csv('news.csv')

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
df.to_csv('predictions_testwef.csv', index=False)


input_text = "no comment is expected from barack obama members of the fyf911 or fukyoflag and blacklivesmatter movements called for the lynching and hanging of white people and cops they encouraged others on a radio show tuesday night to turn the tide and kill white people and cops to send a message about the killing of black people in america one of the f yoflag organizers is called sunshine she has a radio blog show hosted from texas called sunshine s f ing opinion radio show a snapshot of her fyf911 lolatwhitefear twitter page at 9 53 p m shows that she was urging supporters to call now fyf911 tonight we continue to dismantle the illusion of white below is a snapshot twitter radio call invite fyf911the radio show aired at 10 00 p m eastern standard time during the show callers clearly call for lynching and killing of white people a 2 39 minute clip from the radio show can be heard here it was provided to breitbart texas by someone who would like to be referred to as hannibal he has already received death threats as a result of interrupting fyf911 conference calls an unidentified black man said when those mother f kers are by themselves that s when when we should start f ing them up like they do us when a bunch of them ni ers takin one of us out that s how we should roll up he said cause we already roll up in gangs anyway there should be six or seven black mother f ckers see that white person and then lynch their ass let s turn the tables they conspired that if cops started losing people then there will be a state of emergency he speculated that one of two things would happen a big ass r s war or ni ers they are going to start backin up we are already getting killed out here so what the f k we got to lose sunshine could be heard saying yep that s true that s so f king true he said we need to turn the tables on them our kids are getting shot out here somebody needs to become a sacrifice on their side he said everybody ain t down for that s t or whatever but like i say everybody has a different position of war he continued because they don t give a f k anyway he said again we might as well utilized them for that s t and turn the tables on these n ers he said that way we can start lookin like we ain t havin that many casualties and there can be more causalities on their side instead of ours they are out their killing black people black lives don t matter that s what those mother f kers so we got to make it matter to them find a mother f ker that is alone snap his ass and then f in hang him from a damn tree take a picture of it and then send it to the mother f kers we just need one example and then people will start watchin this will turn the tables on s t he said he said this will start a trickle down effect he said that when one white person is hung and then they are just flat hanging that will start the trickle down effect he continued black people are good at starting trends he said that was how to get the upper hand another black man spoke up saying they needed to kill cops that are killing us the first black male said that will be the best method right there breitbart texas previously reported how sunshine was upset when racist white people infiltrated and disrupted one of her conference calls she subsequently released the phone number of one of the infiltrators the veteran immediately started receiving threatening calls one of the f yoflag movement supporters allegedly told a veteran who infiltrated their publicly posted conference call we are going to rape and gut your pregnant wife and your f ing piece of sh t unborn creature will be hung from a tree breitbart texas previously encountered sunshine at a sandra bland protest at the waller county jail in texas where she said all white people should be killed she told journalists and photographers you see this nappy ass hair on my head that means i am one of those more militant negroes she said she was at the protest because these redneck mother f kers murdered sandra bland because she had nappy hair like me fyf911 black radicals say they will be holding the imperial powers that are actually responsible for the terrorist attacks on september 11th accountable on that day as reported by breitbart texas there are several websites and twitter handles for the movement palmetto star describes himself as one of the head organizers he said in a youtube video that supporters will be burning their symbols of the illusion of their superiority their false white supremacy like the american flag the british flag police uniforms and ku klux klan hoods sierra mcgrone or nocturnus libertus posted you too can help a young afrikan clean their a with the rag of oppression she posted two photos one that appears to be herself and a photo of a black man wiping their naked butts with the american flag for entire story breitbart news"

inputs = tokenizer(input_text, return_tensors='pt', max_length=32 ,truncation=True)

inputs.to(device)

with torch.no_grad():
    outputs = model(**inputs)

predicted_label = torch.argmax(outputs.logits).item()

predicted_class = label_map[predicted_label]

print("Predicted class:", predicted_class)