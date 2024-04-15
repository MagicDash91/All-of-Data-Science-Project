import streamlit as st
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud
import PyPDF2
import re
from io import StringIO
import plotly.express as px
import pandas as pd
import collections
import seaborn as sns
sns.set_theme(color_codes=True)
import os
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import PIL.Image
import matplotlib.pyplot as plt

st.title("NLP : PDF Document Analysis")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to convert text to Markdown format
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Create stopword remover
stop_factory = StopWordRemoverFactory()
more_stopword = ['dengan', 'ia', 'bahwa', 'oleh', 'rp', 'undang', 'pasal', 'ayat', 'bab']
data = stop_factory.get_stop_words() + more_stopword

# User input for custom stopwords
custom_stopwords = st.text_input("Enter custom stopwords (comma-separated):")
if custom_stopwords:
    custom_stopword_list = [word.strip() for word in custom_stopwords.split(",")]
    data.extend(custom_stopword_list)

# Function to read PDF and return string
def read_pdf(file):
    pdf_reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page).extractText()
    return text

# Upload PDF file
file = st.file_uploader("Upload a PDF file", type="pdf", key='text1')

# If file is uploaded
if file is not None:
    # Call read_pdf function to convert PDF to string
    text1 = read_pdf(file)

    # Stem and preprocess the text
    sentence1 = text1
    output1 = stemmer.stem(sentence1)
    hasil1 = re.sub(r"\d+", "", output1)
    hasil1 = re.sub(r'[^a-zA-Z\s]', '', hasil1)
    pattern = re.compile(r'\b(' + r'|'.join(data) + r')\b\s*')
    hasil1 = pattern.sub('', hasil1)

    # Create WordCloud
    wordcloud = WordCloud(
        min_font_size=3, max_words=200, width=800, height=400,
        colormap='Set2', background_color='white'
    ).generate(hasil1)

    # Save the WordCloud image
    wordcloud_file = "wordcloud.png"
    wordcloud.to_file(wordcloud_file)

    # Display the WordCloud using Streamlit
    st.subheader(f"Wordcloud Visualization")
    st.image(wordcloud_file)

    # Use Google Gemini API to generate content based on the uploaded image
    st.subheader("Google Gemini Response")

    # Load the image
    img = PIL.Image.open(wordcloud_file)

    # Configure and use the GenerativeAI model
    genai.configure(api_key="AIzaSyDU0F3ZmGWBrrFpmUv21ZHuJBoTbtm4mL8")
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
    response.resolve()

    # Display Gemini API response in Markdown format
    st.write(response.text)

    # Use Google Gemini API to generate content based on the WordCloud image
    genai.configure(api_key="AIzaSyDU0F3ZmGWBrrFpmUv21ZHuJBoTbtm4mL8")
    model = genai.GenerativeModel('gemini-pro-vision')
    response_gemini = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
    response_gemini.resolve()

    # Bigram visualization
    # Get bigrams
    words1 = hasil1.split()
    # Get bigrams
    bigrams = list(zip(words1, words1[1:]))

    # Count bigrams
    bigram_counts = collections.Counter(bigrams)

    # Get top 10 bigram counts
    top_bigrams = dict(bigram_counts.most_common(10))

    # Create bar chart
    plt.figure(figsize=(10, 7))
    plt.bar(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
    plt.xticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=90)
    plt.xlabel('Bigram Words')
    plt.ylabel('Count')
    plt.title(f"Top 10 Bigram from PDF Document")

    # Add Gemini response text to the plot
    gemini_response_text = response_gemini.text
   
    # Save the entire plot as a PNG
    plt.tight_layout()
    plt.savefig("bigram_with_gemini_response.png")

    # Display the plot and Gemini response in Streamlit
    st.subheader("Bigram for PDF Document")
    st.image("bigram_with_gemini_response.png")
    st.subheader("Google Gemini Response")
    st.write(gemini_response_text)
