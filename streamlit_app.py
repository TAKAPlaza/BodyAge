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
    st.image(X_test[n], caption=n, use_column_width=True)
    col1, col2 = st.columns(2)
    col1.metric(label="実年齢", value=y_test[n][0])
    col2.metric(label="肉体年齢", value=int(preds[n][0]))
    if (int(preds[n][0]-y_test[n][0]>5):
        st.write("肉体年齢は年齢高く、要注意です。")
    elif (int(preds[n][0]-y_test[n][0]<-5):
        st.write("肉体年齢は年齢より若く、健康的です。")
    else: st.write("肉体年齢は年齢相応です。")