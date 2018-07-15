FROM python:3.6-slim

ENV HOME=/app

WORKDIR $HOME

COPY requirements.txt $HOME/

RUN pip install -r requirements.txt

COPY . $HOME/

RUN pip install -e .

ENV FLASK_APP=$HOME/http_wrapper/app.py
CMD ["flask", "run", "--host=0.0.0.0"]
