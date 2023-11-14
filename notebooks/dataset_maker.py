from pytube import Playlist
from pytube.exceptions import AgeRestrictedError
import pandas as pd
from string import ascii_letters, digits, whitespace
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count
from os import mkdir
from os.path import exists

num_cores = cpu_count()
num_cores = int(num_cores/2)
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

def main():
    parser = argparse.ArgumentParser(description='Download playlists from a CSV file.')
    parser.add_argument('csv_loc', type=str, help='The location of the CSV file.')
    parser.add_argument('save_path', type=str, help='The path where the playlists will be saved.')
    args = parser.parse_args()

    download_playlists(args.csv_loc, args.save_path)


if __name__ == '__main__':
    main()
