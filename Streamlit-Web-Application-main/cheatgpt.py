import streamlit as st
from PIL import Image
import io
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown

st.title("CheatGPT")

uploaded_file = st.file_uploader("Upload your PNG or JPG image:", type=["png", "jpg"])

if uploaded_file is not None:

    # Validate the file extension
    if uploaded_file.type in ["image/png", "image/jpeg"]:
        # Read the image bytes
        img_bytes = uploaded_file.read()

        # Convert bytes to PIL Image object
        img = Image.open(io.BytesIO(img_bytes))
        st.write("Image Uploaded")
        st.image(img)

        img.save("image.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyDU0F3ZmGWBrrFpmUv21ZHuJBoTbtm4mL8")

        import PIL.Image

        img1 = PIL.Image.open("image.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["Answer This Question and give the explanation", img1], stream=True)
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)

            