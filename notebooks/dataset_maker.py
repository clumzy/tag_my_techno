from pytube import Playlist
from pytube.exceptions import AgeRestrictedError
import pandas as pd
from string import ascii_letters, digits, whitespace
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count
from os import mkdir
from os.path import exists

num_cores = cpu_count()
legal_chars = ascii_letters + digits + whitespace


def _download_video(package):
    video = package[0]
    loc = package[1]
    title = ("".join([c for c in list(video.title) if c in legal_chars])).replace(" ", "_")
    try:
        video.streams.get_audio_only().download(
            filename=loc+title+".mp3")
    except AgeRestrictedError:
        pass


def download_playlists(
        csv_loc: str,
        save_path: str):
    plists = pd.read_csv(
        filepath_or_buffer=csv_loc,
        sep=",")
    for pl in pd.unique(plists["Genre"]):
        new_path = save_path+"/"+pl
        if not exists(new_path):
            mkdir(path=new_path)
    for pl in plists.iterrows():
        p = Playlist(pl[1]["Playlist"])
        len_pl = len(list(p.videos))
        videos = zip(list(p.videos), [save_path+"/"+pl[1]["Genre"]+"/"]*len_pl)
        print(f"Loading {p.title} :")
        process_map(_download_video, videos, max_workers=num_cores, total=len_pl)


if __name__ == '__main__':
    a = "D:/George/Documents/Code/clumzy/tag_my_techno/datasets/genres.csv"
    b = "D:/George/Documents/Code/clumzy/tag_my_techno/datasets"
    download_playlists(a, b)
