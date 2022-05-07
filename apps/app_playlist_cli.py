from importlib.resources import files
from unittest.util import sorted_list_difference
import tagmytechno as tmt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from p_tqdm import p_map
from pathlib import Path
from tqdm import tqdm
from pprint import pprint

def get_files_list(location = "/home/george/Code/clumzy/tag_my_techno/datasets/audio/"):
    audio_files = np.empty((0,1))
    # LES EXTENSIONS QUE L'ON CHERCHE
    exts = ['*.flac','*.mp3', "*.wav", "*.aiff"]
    audio_files = [path for ext in exts for path in Path(location).rglob(f'{ext}')]
    return audio_files

def predict_and_sort_audio(model, files_list, threshold = 0.85):
    sorted_list = {}
    images = list(p_map(tmt.song_to_img, files_list, num_cpus=6))
    results = [tmt.get_genre_prediction(model, img, threshold) for img in tqdm(images)]
    for tune, result in zip(files_list,results):
        for g in result:
            if g[1] in sorted_list.keys(): sorted_list[g[1]].append(tune)
            else: sorted_list[g[1]] = [tune]
    return sorted_list

def playlist_from_list(m3u_location, audio_list):
    m3u = Path(m3u_location)
    m3u.touch(exist_ok=True)
    f = open(m3u, "a", encoding="utf-8")
    for audio in audio_list:
        f.write(str(audio)+"\n")
    f.close()

model = tmt.create_model("/home/george/Code/clumzy/tag_my_techno/apps/mdl.keras")
files = get_files_list("/media/george/George/music/2203")
lst = predict_and_sort_audio(model, files, 0.95)
for playlist in lst:
    playlist_from_list(os.path.join('/home/george/Code/clumzy/tag_my_techno/apps/example_playlists',f"{playlist.lower().replace(' ', '_')}.m3u"), lst[playlist])