import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from numpy.linalg import norm
import tensorflow
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
import pickle

feature_list = np.array(pickle.load(open('superhero_featured_array.pkl', 'rb')))
filenames = pickle.load(open('superhero_image_paths.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.markdown("<h2 style='text-align: center;'>SUPERHEROS IMAGE RETRIEVAL</h2>", unsafe_allow_html=True)

file = st.file_uploader("Please upload a SUPERHERO image", type=['jpg', 'png'])

if file is None:
    st.text('Please upload an image file')
else:
    size = (224, 224)
    img = Image.open(file)
    img_path = ImageOps.fit(img, size, Image.ANTIALIAS)
    img_arr = np.array(img_path)
    extend_img = np.expand_dims(img_arr, axis=0)
    pre_process_img = preprocess_input(extend_img)
    result = model.predict(pre_process_img).flatten()
    normalized = result / norm(result)
    neighbors = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distance, indices = neighbors.kneighbors([normalized])

    original_image,c1,c2,c3,c4=st.columns(5)
    c5,c6,c7,c8,c9=st.columns(5)
    c10,c11,c12,c13,c14=st.columns(5)
    c15,c16,c17,c18,c19=st.columns(5)
    c20,c21,c22,c23,c24=st.columns(5)
    c25,c26,c27,c28,c29=st.columns(5)
    c30,c31,c32,c33,c34=st.columns(5)

    with original_image:
        st.image(filenames[indices[0][0]],use_column_width=True,caption="Uploaded Image")
    with c1:
        st.image(filenames[indices[0][1]],use_column_width=True)
    with c2:
        st.image(filenames[indices[0][2]],use_column_width=True)
    with c3:
        st.image(filenames[indices[0][3]],use_column_width=True)
    with c4:
        st.image(filenames[indices[0][4]],use_column_width=True)
    with c5:
        st.image(filenames[indices[0][5]],use_column_width=True)
    with c6:
        st.image(filenames[indices[0][6]],use_column_width=True)
    with c7:
        st.image(filenames[indices[0][7]],use_column_width=True)
    with c8:
        st.image(filenames[indices[0][8]],use_column_width=True)
    with c9:
        st.image(filenames[indices[0][9]],use_column_width=True)
    with c10:
        st.image(filenames[indices[0][10]],use_column_width=True)
    with c11:
        st.image(filenames[indices[0][11]],use_column_width=True)
    with c12:
        st.image(filenames[indices[0][12]],use_column_width=True)
    with c13:
        st.image(filenames[indices[0][13]],use_column_width=True)
    with c14:
        st.image(filenames[indices[0][14]],use_column_width=True)
    with c15:
        st.image(filenames[indices[0][15]],use_column_width=True)
    with c16:
        st.image(filenames[indices[0][16]],use_column_width=True)
    with c17:
        st.image(filenames[indices[0][17]],use_column_width=True)
    with c18:
        st.image(filenames[indices[0][18]],use_column_width=True)
    with c19:
        st.image(filenames[indices[0][19]],use_column_width=True)
    with c20:
        st.image(filenames[indices[0][20]],use_column_width=True)
    with c21:
        st.image(filenames[indices[0][21]],use_column_width=True)
    with c22:
        st.image(filenames[indices[0][22]],use_column_width=True)
    with c23:
        st.image(filenames[indices[0][23]],use_column_width=True)
    with c24:
        st.image(filenames[indices[0][24]],use_column_width=True)
    with c25:
        st.image(filenames[indices[0][25]],use_column_width=True)
    with c26:
        st.image(filenames[indices[0][26]],use_column_width=True)
    with c27:
        st.image(filenames[indices[0][27]],use_column_width=True)
    with c28:
        st.image(filenames[indices[0][28]],use_column_width=True)
    with c29:
        st.image(filenames[indices[0][29]],use_column_width=True)
    with c30:
        st.image(filenames[indices[0][30]],use_column_width=True)
    with c31:
        st.image(filenames[indices[0][31]],use_column_width=True)
    with c32:
        st.image(filenames[indices[0][32]],use_column_width=True)
    with c33:
        st.image(filenames[indices[0][33]],use_column_width=True)
    with c34:
        st.image(filenames[indices[0][34]],use_column_width=True)
    
