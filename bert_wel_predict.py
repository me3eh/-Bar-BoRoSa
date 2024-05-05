import mlflow.pytorch
from transformers import BertTokenizer
import torch
from dotenv import load_dotenv
import os
load_dotenv(".env")

mlflow.set_tracking_uri(os.environ.get("MLFLOW_SERVER"))

logged_model = 'runs:/3a33c32202da47999ac3c474edad9618/model'

device = torch.device('mps')

# Load model as a PyFuncModel.
model = mlflow.pytorch.load_model(logged_model)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input text
textReal = """
President Joe Biden on Wednesday called close US ally Japan “xenophobic” at a Washington, D.C., fundraiser, just weeks after lauding the US-Japan alliance at a state dinner.

The president made the remark at the off-camera event while arguing that Japan, along with India, Russia and China, would perform better economically if the countries embraced immigration more.

“You know, one of the reasons why our economy is growing is because of you and many others. Why? Because we welcome immigrants. We look to – the reason – look, think about it – why is China stalling so badly economically? Why is Japan having trouble? Why is Russia? Why is India? Because they’re xenophobic. They don’t want immigrants,” Biden said, according to an official White House transcript released Thursday. An initial report of Biden’s comments that was released by pool reporters did not include India in the list of countries he mentioned.

On Thursday, press secretary Karine Jean-Pierre said the president was attempting to make a larger point when he described Japan and India as “xenophobic.”

“He was saying that when it comes to who we are as a nation, we are a nation of immigrants, that is in our DNA,” she told reporters aboard Air Force One, adding later Biden was making a “broad comment” in his comments about Japan and India.

She described the US-Japan relationship as “important” and “enduring” that would continue, despite Biden’s comment. As for whether the president would make similar remarks going forward, she said: “That is up to the president.”

Earlier in the day, National Security Council spokesman John Kirby said he wasn’t aware of any communications between the White House and the governments of Japan or India.

“President Biden values the capabilities that they bring across the spectrum on a range of issues, not just security related,” Kirby said.

Biden had similarly cast Japan, Russia and China as “xenophobic” during an interview with a Spanish language radio station in March.

“The Japanese, the Chinese, they’re xenophobic, they don’t want any – the Russians, they don’t want to have people, other than Russians, Chinese, or Japanese,” the president said at the time.

The latest critique of Japan comes less than a month after he hosted Japanese Prime Minister Fumio Kishida for a state visit and nearly a year after the president hosted Indian Prime Minister Narendra Modi for his own state visit. Biden has leaned on improving relations with both Japan and India as important counterweights to China’s growing global influence.

At the state dinner held at the White House in April, Biden said Japan and the US share “the same values, the same commitment to democracy and freedom to dignity.”

“And today without question, our alliance is literally stronger than it has ever been,” Biden said during the dinner.

Japan has long experienced a demographic crisis with far-reaching consequences for the country’s workforce and economy. Japan and other East Asian nations have largely shied away from using immigration to bolster their populations.

The president’s comments also come as he’s facing political pressure at home over his own immigration policies amid strained resources to deal with an influx of migrants and sharp Republicans criticism.

This story has been updated with a quote from the official White House transcript and additional reporting.
"""

text = """
Top Trump Surrogate BRUTALLY Stabs Him In The Back: ‘He’s Pathetic’ (VIDEO) It s looking as though Republican presidential candidate Donald Trump is losing support even from within his own ranks. You know things are getting bad when even your top surrogates start turning against you, which is exactly what just happened on Fox News when Newt Gingrich called Trump  pathetic. Gingrich knows that Trump needs to keep his focus on Hillary Clinton if he even remotely wants to have a chance at defeating her. However, Trump has hurt feelings because many Republicans don t support his sexual assault against women have turned against him, including House Speaker Paul Ryan (R-WI). So, that has made Trump lash out as his own party.Gingrich said on Fox News: Look, first of all, let me just say about Trump, who I admire and I ve tried to help as much as I can. There s a big Trump and a little Trump. The little Trump is frankly pathetic. I mean, he s mad over not getting a phone call? Trump s referring to the fact that Paul Ryan didn t call to congratulate him after the debate. Probably because he didn t win despite what Trump s ego tells him.Gingrich also added: Donald Trump has one opponent. Her name is Hillary Clinton. Her name is not Paul Ryan. It s not anybody else. Trump doesn t seem to realize that the person he should be mad at is himself because he truly is his own worst enemy. This will ultimately lead to his defeat and he will have no one to blame but himself.Watch here via Politico:Featured Photo by Joe Raedle/Getty Images
"""

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

# Print the predicted class
print(predicted_class)
