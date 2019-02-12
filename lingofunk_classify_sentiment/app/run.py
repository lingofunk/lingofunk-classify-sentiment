import argparse
import logging
import sys

from flask import Flask, Response, jsonify, render_template, request

from lingofunk_classify_sentiment.classify import Classifier
from lingofunk_classify_sentiment.config import config, fetch
from lingofunk_classify_sentiment.data.city_sentiment_analyst import CitySentimentAnalyst

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Server:
    def __init__(self, app: Flask, analyst: CitySentimentAnalyst, port: int):
        self._app = app
        self._port = port
        self._analyst = analyst

        # routes
        app.route("/activations", methods=["GET", "POST"])(self.activations)
        app.route("/histogram", methods=["GET", "POST"])(self.histogram)

    def activations(self):
        """
        Receive a text and return the activation map
        """
        if request.method == "POST":
            logger.debug(request.get_json())
            data = request.get_json()
            text = data.get("text", "")
            if len(text.strip()) == 0:
                return Response(status=400)
            processed_text = self._analyst.classifier.preprocess(text)
            activation_maps = self._analyst.classifier.activation_maps(text)
            prediction = self._analyst.classifier.classify(text)
            data = {
                "activations": activation_maps,
                "processed_text": processed_text,
                "prediction": prediction,
            }
            return jsonify(data)
        else:
            return Response(status=501)

    def histogram(self):
        """
        Return a list of numbers of restaurants with corresponding sentiments
        """
        if request.method == "POST":
            data = request.get_json()
            business_id = data.get("business_id", "")
            if len(business_id.strip()) == 0:
                return Response(status=400)

            histogram = self._analyst.get_histogram_for_restaurant_name(business_id).tolist()
            return jsonify(histogram=histogram)
        else:
            return Response(status=501)

    def serve(self):
        self._app.run(host="0.0.0.0", port=self._port, debug=True, threaded=True)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="The port to listen to (the default is 8000).",
    )
    return parser.parse_args()


def run():
    args = load_args()
    app = Flask(__name__)
    classifier = Classifier()
    server = Server(app, classifier, args.port)
    server.serve()


if __name__ == "__main__":
    run()
