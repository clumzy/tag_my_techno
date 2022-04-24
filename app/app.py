import genre_finder as genre_finder

import streamlit as st

import os
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Electronic music genre finder using a custom CNN Deep Neural Network.")

st.header("Part 1: File upload")
uploaded_file = st.file_uploader(
    "Choose a file", 
    type=['mp3','flac','wav'],
    accept_multiple_files=False)
if uploaded_file is not None:
    st.success("File upload succesful.")
    model = genre_finder.load_model("mdl_wts.hdf5")
    st.header("Part 2: Image from audio file")
    song_img = genre_finder.song_to_img(uploaded_file)
    st.image(song_img)
    if st.button("Start inference :"):
        st.header("Part 3: Genre inference")
        results = [g for g in genre_finder.get_genre_prediction(model, song_img) if g[0] > 0.8]
        fig = plt.figure(figsize = (10,5))
        genre_plot = sns.barplot(x = [x[1] for x in results], y = [x[0] for x in results], palette="rocket")
        genre_plot.bar_label(genre_plot.containers[0], fmt='%.3f')
        plt.title(uploaded_file.name+": detected genres.")
        plt.xticks(rotation=45)
        st.pyplot(fig)