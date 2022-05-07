import genre_finder as genre_finder

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

#VERSION 101

st.set_page_config(page_title="🎵 Tag My Techno 🎵") #type: ignore
st.title("🎵 Tag My Techno 🎵")
st.header("An electronic music genres detector using a custom CNN Deep Neural Network")
with st.spinner('Loading model...'):
    model = genre_finder.create_model("mdl.keras")
st.header("Part 1: File upload")
uploaded_file = st.file_uploader(
    "Please upload an audio file", 
    type=['mp3','flac','wav'],
    accept_multiple_files=False)
if uploaded_file is not None:
    st.success("File upload succesful.")
    st.header("Part 2: Image from audio file")
    with st.spinner('Extracting features...'):
        song_img = genre_finder.song_to_img(uploaded_file)
        r,g,b = genre_finder.split_rgb(song_img)
    st.markdown(body = "### Feature 1: Constant-Q aka *intensity*")
    st.image(r, output_format = "PNG")
    st.markdown(body = "### Feature 2: MFCC aka *perceived frequencies*")
    st.image(g, output_format = "PNG")
    st.markdown(body = "### Feature 3: Chromagram (CENS Variant) aka *melody*")
    st.image(b, output_format = "PNG")
    st.markdown(body = "### Which all adds up to this picture :")
    st.image(song_img, output_format = "PNG")
    st.header("Part 3: Genre inference")
    with st.spinner('Loading predictions...'):
        results = [g for g in genre_finder.get_genre_prediction(model, song_img) if g[0] >= 0.85]
        fig = plt.figure(figsize = (10,4))
        sns.set_style(style='whitegrid')
        genre_plot = sns.barplot(x = [x[1] for x in results], y = [x[0] for x in results], palette="rocket")
        genre_plot.bar_label(genre_plot.containers[0], fmt='%.3f')
        plt.title(uploaded_file.name+": the following genres have been detected !")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
    st.header("Part 4: Inference evaluation")
    st.text("Did the model guess the genre properly ?")
    with st.form("send_feedback"):
        options = st.multiselect(
            'Please select all the current genres that apply to this audio file so that we can keep on improving the model 😊',
            genre_finder.genres,
            [x[1] for x in results])
        submitted = st.form_submit_button("Send feedback")
        if submitted:
            st.write("Thank you for sending your feedback !")
