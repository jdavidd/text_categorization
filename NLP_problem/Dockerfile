FROM python:3.8

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

ENV PYTHONPATH /code

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./categorization /code/categorization

CMD ["uvicorn", "categorization.inference:app", "--host", "0.0.0.0", "--port", "80"]