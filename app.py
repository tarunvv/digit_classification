import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO


st.markdown(f'''
<h1>Digit Classification System</h1>
''',unsafe_allow_html=True)

@st.cache_resource()
def finalised_model():
	model=YOLO('best.pt')
	return model

user_image=st.file_uploader('Upload your file',type=['jng','png','jpeg'])
btn=st.button('Classify')
if btn and user_image is not None:

	bytes_data=user_image.getvalue()
	cv2_img=cv2.imdecode(np.frombuffer(bytes_data, np.uint8),cv2.IMREAD_COLOR)
	model=finalised_model()
	model_img=model(cv2_img)
	names_dict=model_img[0].names
	initial_probs=model_img[0].probs
	final_probs_list=initial_probs.data.tolist()
	final_val=np.argmax(final_probs_list)
	final_pred_val=names_dict[final_val]
	st.image(user_image,use_column_width=True)
	st.markdown(f'''
	<h3 align='center'>Classified Class:{final_pred_val}</h3>''',unsafe_allow_html=True)
	st.balloons()
elif btn and user_image is None:
	st.warning('Please check you image')