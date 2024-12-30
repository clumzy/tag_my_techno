import tagmytechno as tmt
import numpy as np
import os
from p_tqdm import p_map
from pathlib import Path
from tqdm import tqdm
import click


def get_files_list(location="/home/george/Code/clumzy/tag_my_techno/datasets/audio/"):
    audio_files = np.empty((0,1))
    # LES EXTENSIONS QUE L'ON CHERCHE
    exts = ['*.flac','*.mp3', "*.wav", "*.aiff"]
    audio_files = [path for ext in exts for path in Path(location).rglob(f'{ext}')]
    return audio_files


def predict_and_sort_audio(model, files_list, threshold=0.85, num_cpus=4):
    sorted_list = {}
    print(f"Creating images.")
    images = list(p_map(tmt.song_to_img, files_list, num_cpus=num_cpus))
    print(f"Generating genre predictions.")
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


@click.command()
@click.option('--model_loc', default="mdl.keras", help='The location of the model.', show_default=True)
@click.option('--audio_loc', required=True, help='The location of the audio files to guess.')
@click.option('--playlist_loc', required=True, help='The location where you want the playlists to be stored.')
@click.option('--threshold', default=0.95, help="The threshold above which a genre is recognized.", show_default=True)
@click.option('--num_cpus', type=int, default=4, help="Number of CPUs to use for parallel processing.", show_default=True)
def sorter(model_loc, audio_loc, playlist_loc, threshold, num_cpus):
    if not audio_loc:
        raise click.UsageError("Please provide the audio location (--audio_loc)")
    if not playlist_loc:
        raise click.UsageError("Please provide the playlist location (--playlist_loc)")

    model = tmt.create_model(model_loc)
    audio_files = get_files_list(audio_loc)
    print(f"Now analyzing {len(audio_files)} files :")
    lst = predict_and_sort_audio(model, audio_files, threshold, num_cpus)
    print(f"Now creating {len(lst)} playlists.")
    for playlist in lst:
        playlist_from_list(os.path.join(playlist_loc, f"{playlist.lower().replace(' ', '_')}.m3u"), lst[playlist])


if __name__ == '__main__':
    sorter()
