import streamlit as st
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pickle
from PIL import Image

st.title('image classfification ')
st.write('this is an image classification model which can predict 3 diffrent image by using svm or support vector machine SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.')
st.title('you can predict between : ')
st.markdown('1. cerstano ronaldo')
st.markdown('2. motorbike')
st.markdown('3. cars')
model = pickle.load(open('model.p','rb'))
upload_file = st.file_uploader('choose image', type=['png','jpg','jpeg','jfif'])

if upload_file is not None:
    img = Image.open(upload_file)
    st.image(img,caption='sucess')
    if st.button('Predict'):
        st.write('predict........')
        flat_data = []
        img = np.array(img)
        img_resized = resize(img,(150,150,3))
        flat_data.append(img_resized.flatten())
        flat_data = np.array(flat_data)
        y_out = model.predict(flat_data)
        CATEGORY = ['Cars', 'cristiano ronaldo','motorbike']
        y_out = CATEGORY[y_out[0]]
        st.write(f'output is : {y_out}')
        q = model.predict_proba(flat_data)
        for index, item in enumerate(CATEGORY):
            st.write(f'{item} : {q[0][index]*100}%')

