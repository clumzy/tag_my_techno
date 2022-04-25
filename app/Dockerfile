FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1 ffmpeg
COPY . ./app
WORKDIR /app
RUN pip3 install -r requirements.txt
EXPOSE 8080
CMD streamlit run app.py --server.port 8080