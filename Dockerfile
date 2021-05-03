# Tensforlow 2 official image with Python 3.6 as base
#FROM tensorflow/tensorflow:2.0.0-py3
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# Maintainer info

#LABEL maintainer="blahbah@something.com"

# Make working directories
#RUN  mkdir -p /home/project-api
WORKDIR /

ADD . /
# Upgrade pip with no cache


RUN pip install --no-cache-dir -U pip
RUN pip install pipenv
RUN pipenv install --dev
#RUN pipenv shell
# Copy application requirements file to the created working directory
# COPY requirements.txt .

# Install application dependencies from the requirements file
RUN pip install -r requirements.txt

# Copy every file in the source folder to the created working directory
#COPY  . .

#ENTRYPOINT [ "python" ]

EXPOSE 8888

# Run the python application
CMD ["uvicorn", "sentiment_analyzer.api:app", "--host", "0.0.0.0", "--port", "8888"]
#CMD ["./bin/start_server"]