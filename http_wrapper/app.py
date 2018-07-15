''' Flask app for ape '''

import markdown
from flask import Flask, Markup, jsonify, render_template, request

from http_utils import PandasEncoder
from nk_ape import Ape

app = Flask(__name__)
app.json_encoder = PandasEncoder

ape_client = Ape()
print('ape loaded')


@app.route('/')
def homepage():
    with open('README.md') as readme:
        content = readme.read()

    content = Markup(markdown.markdown(content))
    return render_template('index.html', content=content)


@app.route('/concepts/<input_words>')
def req_predictions(input_words):
    result = get_predictions(input_words)
    return jsonify(result)


def get_predictions(input_words):
    if isinstance(input_words, str):
        input_words = input_words.split(',')
    return ape_client.predict_labels(input_words)


if __name__ == "__main__":
    app.run(debug=True)
