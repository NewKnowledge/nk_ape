''' Flask app for ape '''

import os

import markdown
from flask import Flask, Markup, jsonify, render_template, request
from nk_ape import Ape

from http_utils import PandasEncoder

FLASK_ENV = os.getenv('FLASK_ENV', 'development')

app = Flask(__name__)
app.json_encoder = PandasEncoder

ape_client = Ape(verbose=(FLASK_ENV == 'development'))


@app.route('/')
def homepage():
    with open('README.md') as readme:
        content = readme.read()

    content = Markup(markdown.markdown(content))
    return render_template('index.html', content=content)


@app.route('/classes/<input_words>')
def req_predictions(input_words):
    n_classes = int(request.args.get('n_classes', 10))
    separator = request.args.get('separator', ',')
    input_words = input_words.split(separator)

    result = ape_client.get_top_classes(input_words, n_classes=n_classes)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
