from flask import Flask, json, request, jsonify
from flask_cors import CORS, cross_origin
# from model.load_model import load_model, predict
# from dotenv import load_dotenv
import os
import random
# load_dotenv("../env")
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# model, tokenizer, device = load_model()

# example for checking if python flask works ;P
@app.route("/")
@cross_origin()
def hello():
    return "Hello World!"


@cross_origin()
@app.route("/detection", methods=['POST'])
def fake_news_decision():
    # Get data from the request

    # print(request)
    data = request.json
    value_for_checking = data.get("data").strip().lower()
    # print("Value from js")
    # print("'", value_for_checking, "'")
    # Assuming the text data is in the 'text' field of the JSON data
    # text = data.get('text', '')

    # Make prediction
    # predicted_class = predict(text, model, tokenizer, device)
    # if value_for_checking == "lol":
    #     predicted_class = "true"
    # elif value_for_checking == "xD":
    #     predicted_class = "false"
    # else:
    #     predicted_class = "false"
    n = random.randint(0, 1)
    predicted_class = "true" if n == 1 else "false"
    # Prepare response
    response_data = {"result": predicted_class}

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("API_PORT"))
