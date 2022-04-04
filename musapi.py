
import librosa
import numpy as np
from skimage.transform import resize

def rgb_transform(data):
    return ((data+abs(data.min()))/(data+abs(data.min())).max())*255

def song_to_img(file, hop_length=2048,num_sample=6, sample_length=5, sample_rate=44100):
    HOP_LENGTH = hop_length
    NUM_SAMPLE = num_sample
    SAMPLE_LENGTH = sample_length
    SAMPLING_RATE = sample_rate

    song = librosa.load(file, sr=SAMPLING_RATE)[0]

    samples = SAMPLING_RATE*SAMPLE_LENGTH
    song_inter = np.linspace(0,(song.shape[0]-SAMPLING_RATE*6),NUM_SAMPLE).astype(int)
    song_extracts = np.hstack([song[song_inter[i]:song_inter[i]+samples] for i in range(0,NUM_SAMPLE)])
    
    #CALCUL DE LA FFT POUR LES PERCUSSIONS
    stft = librosa.stft(song_extracts, hop_length=HOP_LENGTH)
    harmonic, percussive = librosa.decompose.hpss(stft)
    p = librosa.amplitude_to_db(percussive, ref=np.max) #type: ignore

    #CALCUL DU MFCC, LES FREQUENCES QUE L'ON ENTEND
    mfcc_song = librosa.feature.mfcc(y=song_extracts, n_mfcc=13, sr=SAMPLING_RATE, lifter=512, hop_length=HOP_LENGTH)

    #CALCUL DU CHROMAGRAMME (LES NOTES)
    chroma = librosa.feature.chroma_stft(y=song_extracts, sr=SAMPLING_RATE, hop_length=HOP_LENGTH, center = False)
    r = rgb_transform(resize(p, (646,646), anti_aliasing=None, mode="reflect", order=0)).astype(np.uint)

    g_tempo_bottom = resize(mfcc_song, (646,646), anti_aliasing=None, mode="reflect", order=0)[:int(r.shape[0]/13)+1]
    g_tempo_top = resize(mfcc_song, (646,646), anti_aliasing=None, mode="reflect", order=0)[int(r.shape[0]/13)+1:]

    g = np.vstack([rgb_transform(g_tempo_bottom), rgb_transform(g_tempo_top)]).astype(np.uint)
    b = rgb_transform(resize(chroma, (646,646), anti_aliasing=None, mode="reflect", order=0)).astype(np.uint)
    rgb = np.dstack((r,g,b))
    return rgb