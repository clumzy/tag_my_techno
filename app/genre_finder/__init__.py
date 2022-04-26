import numpy as np

import os
import librosa
import pydub
import librosa.display
from skimage.transform import resize
import warnings
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

from keras.models import load_model

genres = ['Acid House', 'Acid Techno', 'Acid Trance', 'Breakbeat House',
       'Breakbeat Techno', 'Deep House', 'Detroit House',
       'Detroit Techno', 'Ghetto House', 'Hard Techno', 'Hard Trance',
       'Industrial Techno', 'Lofi House', 'Melodic Techno',
       'Minimal Deep Tech', 'Minimal Techno', 'Progressive House',
       'Progressive Trance', 'Psytrance', 'Soulful House', 'Tech House']

def rgb_transform(data):
    """Une fonction qui prend en entrée une image au format Array en RGB, et qui va normaliser
    selon une formule MinMax scalée sur 255.

    Args:
        data (numpy.array): Une image au format numpy Array.

    Returns:
        numpy.array: L'image normalisée.
    """
    return (((data+abs(data.min()))/(data+abs(data.min())).max())*255).astype(np.uint8)

def get_from_pydub(file, normalized=True, num_sample=10, sample_length=3, sample_rate=44100, offset = 0, max_offset = 0):
    """Une fonction qui renvoie un array Numpy représentant un fichier audio découpé selon
    les paramètres indiqués dans la fonction. On anticipe aussi le fait de récupérer plusieurs fois
    le même son, avec la capacité de procéder à un offset. Les différents extraits sont équitablement répartis
    dans le fichier audio.

    Args:
        file (str): L'emplacement du fichier son.
        normalized (bool, optional): Le marqueur de normalisation de l'amplitude du son. Defaults to True.
        num_sample (int, optional): Le nombre de samples pour la découpe. Defaults to 5.
        sample_length (int, optional): La longueur des samples découpés. Defaults to 6.
        sample_rate (int, optional): Le sampling rate du fichier audio. Defaults to 44100.
        offset (int, optional): Le décalage (de sample_length) des différents samples. Defaults to 0.
        max_offset (int, optional): Le nombre maximum de décalages pour ce fichier. Defaults to 2.

    Returns:
        numpy.array: Un array Numpy représentant nos différents extraits.
    """
    NUM_SAMPLE = num_sample
    SAMPLING_RATE = sample_rate
    MAX_OFFSET = max_offset+1
    if MAX_OFFSET-offset <= 0:
        print(f"WARNING, OFFSET {offset} is TOO BIG")
    OFFSET = offset
    song = pydub.AudioSegment.from_file_using_temporary_files(file).set_channels(1)
    #ON REMET LE SAMPLE RATE A 44100 SI CE N'EST PAS LE CAS
    if song.frame_rate != SAMPLING_RATE: song = song.set_frame_rate(sample_rate)
    SAMPLE_LENGTH = sample_length*1000
    # LA CHANSON EST DECOUPEE EN NUM_SAMPLES MORCEAUX, DE LONGEUR SAMPLE_LENGTH(SECONDES)
    song_inter = np.linspace(SAMPLE_LENGTH*(MAX_OFFSET-OFFSET-1)+(1*1000),len(song)-(SAMPLE_LENGTH*(MAX_OFFSET-OFFSET)+(10*1000)),NUM_SAMPLE).astype(int)
    y = np.hstack([song[song_inter[i]:song_inter[i]+SAMPLE_LENGTH].get_array_of_samples() for i in range(0,NUM_SAMPLE)])
    # ON RENVOIE UNE VERSION NORMALISEE DE L'AMPLITUDE
    if normalized:
        return song.frame_rate, np.float32(y) / 2**15
    else:
        return song.frame_rate, y

def song_to_img(file, hop_length=1024, num_sample=10, sample_length=3, sample_rate=44100, offset = 0, max_offset=0):
    """Une fonction qui transforme un fichier audio en image.
    L'image renvoyée est composée du 
        - Constant_Q
        - MFCC (26 bins)
        - Chromagram (36 bins)

    Args:
        file (str): L'emplacement du fichier son.
        hop_length (int, optional): La taille de la fenêtre utilisée pour l'analyse. Defaults to 1024.
        num_sample (int, optional): Le nombre de samples pour la découpe. Defaults to 10.
        sample_length (int, optional): La longueur des samples découpés. Defaults to 3.
        sample_rate (int, optional): Le sampling rate du fichier audio. Defaults to 44100.
        offset (int, optional): Le décalage (de sample_length) des différents samples. Defaults to 0.
        max_offset (int, optional): Le nombre maximum de décalages pour ce fichier. Defaults to 0.

    Returns:
        numpy.array: Une image au format numpy Array, en RGB.
    """
    # VALEUR TEMPORAIRE DE HAUTEUR D'IMAGE, A REVOIR PLUS TARD /!\
    warnings.filterwarnings('ignore')
    HOP_LENGTH = hop_length
    NUM_SAMPLE = num_sample
    SAMPLE_LENGTH = sample_length
    SAMPLING_RATE = sample_rate
    OFFSET = int(offset)
    MAX_OFFSET = int(max_offset)
    pydub_sr, song_extracts = get_from_pydub(
        file, 
        normalized=True,
        num_sample=NUM_SAMPLE,
        sample_length=SAMPLE_LENGTH,
        sample_rate=SAMPLING_RATE,
        offset = OFFSET,
        max_offset = MAX_OFFSET)
    # ON AFFICHE UN AVERTISSEMENT SI LE SAMPLING RATE N'EST PAS BON

    #CALCUL DU CONSTANT Q
    constant_q_temp = librosa.cqt(song_extracts, hop_length=HOP_LENGTH, sr=SAMPLING_RATE)
    constant_q = librosa.amplitude_to_db(np.abs(constant_q_temp))
    
    #CALCUL DU MFCC, LES FREQUENCES QUE L'ON ENTEND
    ## CE FEATURE NE MARCHE PAS TRES BIEN /!\
    mfcc_song = librosa.feature.mfcc(y=song_extracts, n_mfcc=26, sr=SAMPLING_RATE, lifter=512, hop_length=HOP_LENGTH)

    #CALCUL DU CHROMAGRAMME (LES NOTES)
    ## VERSION CENS
    chromacens = librosa.feature.chroma_cens(y=song_extracts, sr=SAMPLING_RATE, hop_length=HOP_LENGTH, n_chroma=36, win_len_smooth=5, C=constant_q_temp)
    #CREATION DE LA SHAPE FINALE A PARTIR DES PLUS GRANDES VALEURS DE NOS TROIS FEATURES
    IM_HEIGHT = max(constant_q.shape[0], mfcc_song.shape[0], chromacens.shape[0])
    IM_WIDTH = max(constant_q.shape[1], mfcc_song.shape[1], chromacens.shape[1])
    IM_SHAPE = (IM_HEIGHT, IM_WIDTH)

    r = rgb_transform(resize(constant_q, (IM_SHAPE), anti_aliasing=None, mode="reflect", order=0)).astype(np.uint)

    g = rgb_transform(resize(mfcc_song, (IM_SHAPE), anti_aliasing=None, mode="reflect", order=0)).astype(np.uint)

    b = rgb_transform(resize(chromacens, (IM_SHAPE), anti_aliasing=None, mode="reflect", order=0)).astype(np.uint)
    
    rgb = np.dstack((r,g,b)).astype(np.uint8)
    return rgb

def split_rgb(song_img):
    """Une fonction qui sépare une image au format numpy Array en 3 images r,g et b.

    Args:
        song_img (numpy.array): Une image au format numpy Array.

    Returns:
        tuple(numpy.array): Trois images R,G et B au format numpy Array.
    """
    one_pad = np.ones(song_img[:,:,0].shape)*255
    r = np.dstack((one_pad,one_pad-song_img[:,:,0],one_pad-song_img[:,:,0])).astype(np.uint8)
    g = np.dstack((one_pad-song_img[:,:,1],one_pad,one_pad-song_img[:,:,1])).astype(np.uint8)
    b = np.dstack((one_pad-song_img[:,:,2],one_pad-song_img[:,:,2],one_pad)).astype(np.uint8)
    return r,g,b

def get_genre_prediction(model, sound):
    """Une fonction qui nous permet de renvoyer les prédictions du genre d'un fichier audio,
    triés par ordre décroissant.

    Args:
        model (keras.models.Sequential): Le modèle utilisé
        sound (numpy.array): L'image représentant le fichier son.

    Returns:
        list(tuple): La liste des prédictions par ordre décroissant.
    """ 
    y_pred = model.predict(np.array([sound])).reshape((21))
    sorted_preds = list(sorted(zip(y_pred,genres), key = lambda x: x[0], reverse = True))
    return sorted_preds

@st.cache(allow_output_mutation=True)
def create_model(model_path):
    """La fonction qui nous permet de recréer le modèle à partir du meilleur checkpoint calculé
    plus tôt.

    Args:
        model_path (str): Le chemin du checkpoint à utiliser.

    Returns:
        keras.models.Sequential: Le modèle avec les poids renseignés.
    """
    my_model = load_model(model_path)
    return my_model
