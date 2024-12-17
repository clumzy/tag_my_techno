# 🎵 Tag My Techno 🎵
*An electronic music genres detector using a custom CNN Deep Neural Network*

## How to try this project

### Use the CLI App available in the repository

This CLI app creates playlists based on electronic music genres detected using a custom Convolutional Neural Network (CNN) model.

#### Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/clumzy/tag_my_techno.git
   ```

2. Navigate to the project directory:
   ```
   cd tag_my_techno/apps
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Place the model in the apps folder (the one you're currently in):

You can download the model here : https://1drv.ms/u/c/0d829c1013c996cb/EcuWyRMQnIIggA3FNQAAAAABw_fTNH4ltZ-_qL685lyjtQ?e=Ostev0

#### Usage

The script is already located in the `apps` directory. To use it, simply run it using Python:

```
python app_playlist_cli.py --audio_loc <path_to_audio_folder> --playlist_loc <path_to_save_playlists>
```

#### Options

The script accepts several options:

- `--model_loc`: Location of the model file (default: "mdl.keras")
- `--audio_loc`: Required. Location of the audio files to analyze
- `--playlist_loc`: Required. Location where playlists will be saved
- `--threshold`: Threshold for genre recognition (default: 0.95)
- `--num_cpus`: Number of CPUs to use for parallel processing (default: 4)

For example:

```
python app_playlist_cli.py --audio_loc /path/to/music/folder --playlist_loc /path/to/save/playlists --threshold 0.90 --num_cpus 8
```

#### How it works

1. The script loads the specified model.
2. It scans the provided audio folder for supported audio files (.flac, .mp3, .wav, .aiff).
3. Each audio file is converted to an image representation.
4. The model predicts genres for each audio file.
5. Files are grouped by predicted genres.
6. Playlists (.m3u files) are created for each detected genre.

#### Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`
- Adequate CPU resources for processing (adjust `--num_cpus` as needed)

#### Troubleshooting

If you encounter any issues:

- Ensure all required dependencies are installed
- Check that the model file is present and accessible
- Verify that the audio folder contains valid audio files
- Adjust the threshold value if too many or too few genres are being detected

#### Contributing

Contributions are welcome! If you'd like to improve the script or add new features, please submit a pull request.

#### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

You can test the classifier yourself thanks to Docker ! More info on the 
link below.


🐳 Docker : https://hub.docker.com/repository/docker/czyclumzy/electronic_tagger
