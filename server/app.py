from flask import Flask, json, request, jsonify
# from model.load_model import load_model, predict
# from dotenv import load_dotenv
import os
# load_dotenv("../env")

app = Flask(__name__)
# model, tokenizer, device = load_model()


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/detection", methods=['POST'])
def fake_news_decision():
    # Get data from the request
    data = request.json
    print("Dupa")
    print(data)
    print(data)
    # Assuming the text data is in the 'text' field of the JSON data
    text = data.get('text', '')

    # Make prediction
    # predicted_class = predict(text, model, tokenizer, device)
    predicted_class = "true"

    # Prepare response
    response_data = {"result": predicted_class}

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("API_PORT"))
