FROM python:3.8

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

ENV PYTHONPATH /code

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet

COPY ./categorization /code/categorization

RUN python /code/categorization/multilabel_knn/setup.py install

CMD ["uvicorn", "categorization.inference:app", "--host", "0.0.0.0", "--port", "80"]