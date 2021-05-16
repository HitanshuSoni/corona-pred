import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.title("CORONA TESTING USING X-RAY")
st.header("Prediction Based on AI")
st.text("Upload a X-RAY of CHEST Image for fast testing of CORONA")

model = tf.keras.models.load_model("x-ray1.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "jpeg", "png"])

map_dict = {0: 'CORONA POSITIVE',
            1: 'CORONA NEGATIVE'
            }

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by Hitanshu Soni || +91-9772418764 || +91-8764004137 || hitanshusoni10@gmail.com</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("{}".format(map_dict [prediction]))

