import genre_finder as genre_finder

import streamlit as st

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

inference_done = False
model = genre_finder.load_model("mdl_wts.hdf5")

st.title("Electronic music genre detector using a custom CNN Deep Neural Network")

st.header("Part 1: File upload")
uploaded_file = st.file_uploader(
    "Choose a file", 
    type=['mp3','flac','wav'],
    accept_multiple_files=False)
if uploaded_file is not None:
    st.success("File upload succesful.")
    st.header("Part 2: Image from audio file")
    song_img = genre_finder.song_to_img(uploaded_file)
    one_pad = np.ones(song_img[:,:,0].shape)*255
    r = np.dstack((one_pad,one_pad-song_img[:,:,0],one_pad-song_img[:,:,0])).astype(np.uint8)
    g = np.dstack((one_pad-song_img[:,:,1],one_pad,one_pad-song_img[:,:,1])).astype(np.uint8)
    b = np.dstack((one_pad-song_img[:,:,2],one_pad-song_img[:,:,2],one_pad)).astype(np.uint8)
    progress_bar = st.progress(0)
    sleep(1)
    progress_bar.progress(25)
    st.markdown(body = "### Feature 1: Constant-Q aka *intensity*")
    st.image(r, output_format = "PNG")
    sleep(1)
    progress_bar.progress(50)
    st.markdown(body = "### Feature 2: MFCC aka *perceived frequencies*")
    st.image(g, output_format = "PNG")
    sleep(1)
    progress_bar.progress(75)
    st.markdown(body = "### Feature 3: Chromagram (CENS Variant) aka *melody*")
    st.image(b, output_format = "PNG")
    sleep(1)
    progress_bar.progress(100)
    st.markdown(body = "### Which all adds up to this picture :")
    st.image(song_img, output_format = "PNG")
    if st.button("Start inference :"):
        st.header("Part 3: Genre inference")
        results = [g for g in genre_finder.get_genre_prediction(model, song_img) if g[0] > 0.8]
        fig = plt.figure(figsize = (10,4))
        sns.set_style(style='whitegrid')
        genre_plot = sns.barplot(x = [x[1] for x in results], y = [x[0] for x in results], palette="rocket")
        genre_plot.bar_label(genre_plot.containers[0], fmt='%.3f')
        plt.title(uploaded_file.name+": detected genres.")
        plt.xticks(rotation=0)
        plt.tight_layout(pad = 1000)
        st.pyplot(fig)
        st.text("Genre inference done !")
        st.header("Part 4: Inference evaluation")
        st.text("Did the model guess the genre properly ?")
