import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

import tensorflow.keras as keras
from keras.models import load_model

import pickle

st.title("あなたの肉体年齢はいくつ？")

# button
col, = st.columns(1)
predict_button = col.button('肉体年齢を計測')

#pickle読み書き
def dumpPickle(fileName, obj):
    with open(fileName, mode="wb") as f:
        pickle.dump(obj, f)
def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)

X_test = loadPickle('X_test.pickle')
y_test = loadPickle('y_test.pickle')

model = load_model('ResNet.hdf5')

preds = model.predict(X_test)

n = random.randint(0, 127)

if predict_button:
    st.image(X_test[n], use_column_width=True)
    st.markdown("あなたの実年齢は**", y_test[n][0], "**歳です。")
    st.markdown("あなたの予測年齢は**"", int(preds[n][0]), "**歳です。")

# if predict_button:
    # fig, axs = plt.subplots(3,10, figsize=(16, 6))
    # axs = axs.flatten()
    # for true, pred, img, ax in zip(y_test, preds, X_test, axs):
    #     pred = round(pred[0],1)
    #     color = 'black' if abs(pred-true)<10 else 'red'
    #     ax.set_title(str(true) + '--' + str(pred), color=color)
    #     ax.axis('off')
    #     ax.imshow(img)
    # plt.show()
