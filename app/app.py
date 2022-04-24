import genre_finder as genre_finder

model = genre_finder.load_model("/home/george/code/clumzy/electronic_tagger/app/mdl_wts.hdf5")
results = genre_finder.get_genre_prediction(model, "/home/george/code/clumzy/electronic_tagger/neonraver.mp3")
print(results)