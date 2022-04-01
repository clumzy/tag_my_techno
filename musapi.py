import pandas as pd
import numpy as np
import audio_metadata
from glob import glob
import requests

class AudioLibrary():
    def __init__(self, music_location:str) -> None:
        self._music_location = music_location
        self._music_files = glob(self._music_location+"/*.mp3")+glob(self._music_location+"/*.flac")+glob(self._music_location+"/*.wav")
        pass   

    def get_search_tags(self, index:int)->dict:
        metadata = audio_metadata.load(self._music_files[index]) #type:ignore
        artist = metadata["tags"]["artist"][0]
        album = metadata["tags"]["album"][0]
        title = metadata["tags"]["title"][0]
        return {
            "artist":artist,
            "album":album,
            "title":title}