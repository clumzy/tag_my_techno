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

import tensorflow as tf
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, BatchNormalization


genres = ['acid_house', 'acid_techno', 'acid_trance', 'breakbeat_house',
       'breakbeat_techno', 'deep_house', 'detroit_house',
       'detroit_techno', 'ghetto_house', 'hard_techno', 'hard_trance',
       'industrial_techno', 'lofi_house', 'melodic_techno',
       'minimal_deep_tech', 'minimal_techno', 'progressive_house',
       'progressive_trance', 'psytrance', 'soulful_house', 'tech_house']

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

@st.cache
def lenet5():
    """Le modèle qui nous permet d'inférer le genre d'un fichier audio.

    Returns:
        keras.models.Sequential: Le modèle.
    """
    model = Sequential()
    model.add(BatchNormalization(input_shape=(84, 1292, 3), momentum=0.99))
    # CONV POOL 0
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization(momentum=0.99))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # CONV POOL 1
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization(momentum=0.99))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # CONV POOL 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization(momentum=0.99))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # CONV POOL 3
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization(momentum=0.99))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # DROPOUT AVANT LE FLATTEN
    model.add(Flatten())
    model.add(BatchNormalization(momentum=0.99))
    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Dense(units=128, activation='relu'))
    # OUTPUT LAYER
    model.add(Dense(units=21, activation = 'sigmoid'))
    
    return model

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
def load_model(checkpoint_path):
    """La fonction qui nous permet de charger le modèle à partir du meilleur checkpoint calculé
    plus tôt.

    Args:
        checkpoint_path (str): Le chemin du checkpoint à utiliser.

    Returns:
        keras.models.Sequential: Le modèle avec les poids renseignés.
    """
    my_model = lenet5()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    my_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    my_model.load_weights(checkpoint_path)
    return my_model
