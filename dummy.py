import os

import streamlit as st
from PIL import Image

from extract_feature import extract_features
from initialize import feature_list, filenames, new_model
from recomendation import recommend
from upload_image import save_uploaded_file

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
#jika tidak kosong
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        #display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        #feature extract
        features = extract_features(os.path.join('uploads',uploaded_file.name),new_model)
        st.text(features.shape)
        #recomendation
        indices = recommend(features.squeeze(),feature_list.squeeze())
        #show
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")