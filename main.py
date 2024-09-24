from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from playsound import playsound
import tensorflow as tf
# print(tf.__version__)
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

from sklearn import metrics
from keras.callbacks import ModelCheckpoint
from datetime import datetime

import librosa  # sound analysis
audio_file_paths = "UrbanSound8k/test/car-horn-6408.mp3"
librosa_audio_data, librosa_sample_rate = librosa.load(audio_file_paths)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(librosa_audio_data)
plt.show()

# voice control
playsound("UrbanSound8k/test/car-horn-6408.mp3")

# Mel Frekans Cepstral Katsayıları/öznitelik haritası çıkarımı
mfccs = librosa.feature.mfcc(
    y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40)

# path
audio_dataset_path = "UrbanSound8k/audio/"
metadata = pd.read_csv("UrbanSound8k/metadata/UrbanSound8K.csv")

# Feature extraction


def feature_extractor(file):
    audio, sample_rate = librosa.load(file, res_type="kaiser_fast")
    mffcs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mffcs_scaled_features = np.mean(mffcs_features.T, axis=0)
    return mffcs_scaled_features


extracted_features = []
for index_num, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(
        audio_dataset_path), "fold"+str(row["fold"])+"/", str(row["slice_file_name"]))
    final_class_labels = row["class"]
    data = feature_extractor(file_name)
    extracted_features.append([data, final_class_labels])

extracted_features_df = pd.DataFrame(
    extracted_features, columns=["features", "class"])

x = np.array(extracted_features_df["features"].tolist())
y = np.array(extracted_features_df["class"].tolist())

le = LabelEncoder()
y = to_categorical(le.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

output_labels = 10


model = Sequential()
model.add(Dense(125, input_shape=((40,)), activation="relu"))
model.add(Dropout(0.5))
# second hidden layer
model.add(Dense(250, activation="relu"))
model.add(Dropout(0.5))
# third hidden layer
model.add(Dense(125, activation="relu"))
model.add(Dropout(0.5))
# output layer
model.add(Dense(output_labels, activation="softmax"))
# model.add(Activation("softmax"))

model.summary()
model.compile(loss="categorical_crossentropy",
              metrics=["accuracy"], optimizer="adam")
epoch = 300
batch_sizee = 32
model.fit(x_train, y_train, batch_size=batch_sizee, epochs=epoch,
          validation_data=(x_test, y_test), verbose=1)

validation_test_set_accuracy = model.evaluate(x_test,y_test,verbose=0)
print(validation_test_set_accuracy)

predict_file = "UrbanSound8k/test/sokak_müziği.mp3"
sound_signal,sample_rate = librosa.load(predict_file,res_type="kaiser_fast")
mfcc_features = librosa.feature.mfcc(y=sound_signal,sr=sample_rate,n_mfcc=40)
mfccs_scaled_features = np.mean(mfcc_features.T,axis=0)

mfccs_scaled_features = mfccs_scaled_features.reshape(-1,1)

# mfccs_scaled_features'ı bir liste içine al
mfccs_scaled_features = np.array(mfccs_scaled_features).reshape(1, -1)  # (1, 40) şekline getir

# Tahmin yap
predict_voice = model.predict(mfccs_scaled_features)
print(predict_voice)

result_classes = ["klima","korna","çocuk sesleri","köpek havlaması","sondaj","motor sesi","silah sesi",
                  "darbeli kırıcı","siren","sokak müziği"]

result = np.argmax(predict_voice)

print("Result = "+ result_classes[result])

if result_classes[result] == "köpek havlaması":
    import matplotlib.image as mpimg
    plt.figure(figsize=(12,4))
    img = mpimg.imread("dog.jpeg")
    imgplot = plt.imshow(img)
    plt.title("Dog Voice")
    plt.show()
    
if result_classes[result] == "klima":
    import matplotlib.image as mpimg
    plt.figure(figsize=(12,4))
    img = mpimg.imread("klima.jpg")
    imgplot = plt.imshow(img)
    plt.title("klima Voice")
    plt.show()

if result_classes[result] == "korna":
    import matplotlib.image as mpimg
    plt.figure(figsize=(12,4))
    img = mpimg.imread("horn.jpeg")
    imgplot = plt.imshow(img)
    plt.title("horn Voice")
    plt.show()

if result_classes[result] == "sondaj":
    import matplotlib.image as mpimg
    plt.figure(figsize=(12,4))
    img = mpimg.imread("drilll.jpg")
    imgplot = plt.imshow(img)
    plt.title("drilll Voice")
    plt.show()
    
if result_classes[result] == "çocuk sesleri":
    import matplotlib.image as mpimg
    plt.figure(figsize=(12,4))
    img = mpimg.imread("children.jpg")
    imgplot = plt.imshow(img)
    plt.title("children Voice")
    plt.show()
    
if result_classes[result] == "motor sesi":
    import matplotlib.image as mpimg
    plt.figure(figsize=(12,4))
    img = mpimg.imread("engine.jpg")
    imgplot = plt.imshow(img)
    plt.title("engine Voice")
    plt.show()

if result_classes[result] == "silah sesi":
    import matplotlib.image as mpimg
    plt.figure(figsize=(12,4))
    img = mpimg.imread("gunshot.jpg")
    imgplot = plt.imshow(img)
    plt.title("gunshot Voice")
    plt.show()
    
if result_classes[result] == "darbeli kırıcı":
    import matplotlib.image as mpimg
    plt.figure(figsize=(12,4))
    img = mpimg.imread("jackhammer.jpg")
    imgplot = plt.imshow(img)
    plt.title("jackhammer Voice")
    plt.show()

if result_classes[result] == "siren":
    import matplotlib.image as mpimg
    plt.figure(figsize=(12,4))
    img = mpimg.imread("siren.jpg")
    imgplot = plt.imshow(img)
    plt.title("siren Voice")
    plt.show()
    
if result_classes[result] == "sokak müziği":
    import matplotlib.image as mpimg
    plt.figure(figsize=(12,4))
    img = mpimg.imread("street_music.jpeg")
    imgplot = plt.imshow(img)
    plt.title("streetmusic Voice")
    plt.show()

