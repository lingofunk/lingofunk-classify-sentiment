from flask import Flask, Response, jsonify, render_template, request

from lingofunk_classify_sentiment.classify import Classifier
from lingofunk_classify_sentiment.config import config, fetch

app = Flask(__name__)

classifier = Classifier(config["models"]["current"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/activations")
def activations():
    """
    Receive a text and return the activation map
    """
    if request.method == "GET":
        text = request.args.get("text", "")
        if len(text.strip()) == 0:
            return Response(status=400)
        processed_text = classifier.preprocess(text)
        activation_maps = classifier.activation_maps(text)
        prediction = classifier.classify(text)
        data = {
            "activations": activation_maps,
            "processed_text": processed_text,
            "prediction": prediction,
            "binary": True
        }
        return jsonify(data)
    else:
        return Response(status=501)

if __name__ == "__main__":
    app.run(port=8090)