# Flask App
from flask import Flask, render_template, jsonify, request

from chat import get_response

app = Flask(__name__)


@app.get("/")  # New Syntax
# Old Syntax -> @app.route("/", methods=["GET"])
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # Check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=True)
