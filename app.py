from flask import Flask, json
import random
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
random.seed(10)

@app.route("/decision")
def fake_news_decision():
    random_number = random.randint(1, 2)
    outcome = True if random_number == 1 else False

    response = app.response_class(
        response= json.dumps({"result": outcome}),
        status=200,
        mimetype='application/json'
    )
    return response