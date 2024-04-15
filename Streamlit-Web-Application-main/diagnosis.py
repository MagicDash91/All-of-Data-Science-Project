import streamlit as st
import datetime
import os
import PIL.Image
import google.generativeai as genai
from IPython.display import Markdown
import time
import io
from PIL import Image
import textwrap

# Replace with your GenerativeAI API key
genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

st.title("CT Scan and MRI Diagnosis Explanator")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    if uploaded_file.type in ["image/png", "image/jpeg"]:
        img_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(img_bytes))
        st.write("Image Uploaded")
        st.image(img)

        img.save("image.png")

        def to_markdown(text):  # Consider removing if formatting not needed
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        model = genai.GenerativeModel('gemini-pro-vision')  # Check supported models
        response = model.generate_content(["Can you analyze this CT scan or MRI and explain any potential abnormalities?", img], stream=True)
        response.resolve()

        st.write("**Google Gemini Response About the image**")
        

        # Extract text from all candidates (GitHub solution)
        text_parts = []
        for candidate in response.candidates:
            text_parts.extend([part.text for part in candidate.content.parts])
        full_text = ''.join(text_parts)  # Join text parts for a cohesive response

        st.write(full_text)  # Display the combined text





