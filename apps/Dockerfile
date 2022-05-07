FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
RUN apt-get install -y --no-install-recommends build-essential gcc libsndfile1 ffmpeg
COPY . ./streamlit
WORKDIR /streamlit
RUN pip3 install -r requirements.txt
EXPOSE 8080
CMD streamlit run app_demo_streamlit.py --server.port 8080