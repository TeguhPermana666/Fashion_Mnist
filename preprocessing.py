import os
import pickle

filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

print(filenames)
# pickle.dump(filenames,open('filenames.pkl','wb'))