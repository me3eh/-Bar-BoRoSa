from flask import Flask, json, request, jsonify
from flask_cors import CORS, cross_origin

from model.load_model import load_model, predict
import os
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = load_model()


@app.route("/")
@cross_origin()
def hello():
    return "Hello World!"

@cross_origin()
@app.route("/detection", methods=['POST'])
def fake_news_decision():
    # Get data from the request
    data = request.json
    value_for_checking = data.get("data").strip().lower()

    # Make prediction
    predicted_class = predict(value_for_checking, model)
    response_data = {"result": predicted_class}

    response = jsonify(response_data)
    return response


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("API_PORT"))
