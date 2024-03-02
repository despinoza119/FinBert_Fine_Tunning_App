# TBD
FROM python:3.11-slim

WORKDIR /usr/src/app

COPY ./ /usr/src/app

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["sh", "-c", "python generate_model.py && streamlit run application.py"]

