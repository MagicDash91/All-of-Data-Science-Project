import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import PyPDF2
sns.set_theme(color_codes=True)
import pandas as pd
from io import StringIO
import re
import os
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

st.title("PDF Document Comparison")

additional_stopwords = st.text_input("Enter additional stopwords (comma-separated)", value="")
additional_stopwords = additional_stopwords.split(",")

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
stop_factory = StopWordRemoverFactory()
more_stopword = ['dengan', 'ia','bahwa','oleh','rp','undang','pasal','ayat','bab']
data = stop_factory.get_stop_words()+more_stopword + additional_stopwords
stopword = stop_factory.create_stop_word_remover()

# Function to read PDF and return string
def read_pdf(file):
    # Create a PyPDF2 reader object
    pdf_reader = PyPDF2.PdfFileReader(file)

    # Extract text from all pages of PDF
    text = ""
    for page in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page).extractText()

    # Return the text as a string
    return text

# Upload PDF file
file = st.file_uploader("Upload a PDF file", type="pdf", key='text1')

# If file is uploaded
if file is not None:
    # Call read_pdf function to convert PDF to string
    text1 = read_pdf(file)


# Function to read PDF and return string
def read_pdf(file):
    # Create a PyPDF2 reader object
    pdf_reader = PyPDF2.PdfFileReader(file)

    # Extract text from all pages of PDF
    text = ""
    for page in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page).extractText()

    # Return the text as a string
    return text

# Upload PDF file
file = st.file_uploader("Upload a PDF file", type="pdf", key='text2')

# If file is uploaded
if file is not None:
    # Call read_pdf function to convert PDF to string
    text2 = read_pdf(file)


if st.button("Process"):

    sentence1 = text1
    output1   = stemmer.stem(sentence1)

    hasil1 = re.sub(r"\d+", "", output1)
    hasil1 = re.sub(r'[^a-zA-Z\s]','',output1)

    pattern = re.compile(r'\b(' + r'|'.join(data) + r')\b\s*')
    hasil1 = pattern.sub('', hasil1)


    sentence2 = text2
    output2   = stemmer.stem(sentence2)

    hasil2 = re.sub(r"\d+", "", output2)
    hasil2 = re.sub(r'[^a-zA-Z\s]','',output2)

    pattern = re.compile(r'\b(' + r'|'.join(data) + r')\b\s*')
    hasil2 = pattern.sub('', hasil2)

    documents = [hasil1, hasil2]
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    # Create the Document Term Matrix
    count_vectorizer = CountVectorizer(stop_words='english')
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(documents)
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim = cosine_similarity(sparse_matrix, sparse_matrix)


    plt.rcParams.update({'font.size': 26})

    heatmap = plt.figure(figsize =(5, 5))
    sns.heatmap(cosine_sim, fmt='.2g', annot=True)


    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    # Create a WordCloud object
    wordcloud = WordCloud(min_font_size=3,max_words=200,width=1600,height=720,
                       colormap = 'Set2', background_color='white').generate(hasil1)

    # Display the WordCloud using Matplotlib and Streamlit
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')


    # Create a WordCloud object
    wordcloud = WordCloud(min_font_size=3,max_words=200,width=1600,height=720,
                       colormap = 'Set2', background_color='white').generate(hasil2)

    # Display the WordCloud using Matplotlib and Streamlit
    fig2, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')


    str=hasil1+hasil2
    # Create a WordCloud object
    wordcloud = WordCloud(min_font_size=3,max_words=200,width=1600,height=720,
                       colormap = 'Set2', background_color='white').generate(str)

    # Display the WordCloud using Matplotlib and Streamlit
    fig3, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')



    #bigram visualization
    import collections
    # Get bigrams
    words1 = hasil1.split()
    bigrams = list(zip(words1, words1[1:]))

    # Count bigrams
    bigram_counts = collections.Counter(bigrams)

    # Get top 10 bigram counts
    top_bigrams = dict(bigram_counts.most_common(10))

    # Create bar chart
    plt.rcParams.update({'font.size': 12})
    fig4, ax = plt.subplots()
    ax.bar(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
    ax.set_xticks(range(len(top_bigrams)))
    ax.set_xticklabels(list(top_bigrams.keys()))
    ax.set_xlabel('Bigram Words')
    ax.set_ylabel('Count')
    ax.set_title('Top 10 Bigram Word Counts')
    plt.xticks(rotation=90)
    plt.figure(figsize =(15, 15))
    



    #bigram visualization
    import collections
    # Get bigrams
    words2 = hasil2.split()
    bigrams = list(zip(words2, words2[1:]))

    # Count bigrams
    bigram_counts = collections.Counter(bigrams)

    # Get top 10 bigram counts
    top_bigrams = dict(bigram_counts.most_common(10))

    # Create bar chart
    plt.rcParams.update({'font.size': 12})
    fig5, ax = plt.subplots()
    ax.bar(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
    ax.set_xticks(range(len(top_bigrams)))
    ax.set_xticklabels(list(top_bigrams.keys()))
    ax.set_xlabel('Bigram Words')
    ax.set_ylabel('Count')
    ax.set_title('Top 10 Bigram Word Counts')
    plt.xticks(rotation=90)
    plt.figure(figsize =(15, 15))

    st.write("**Accuracy**")
    st.write(heatmap)

    st.write("**WordCloud Document 1**")
    st.pyplot(fig)

    st.write("**WordCloud Document 2**")
    st.pyplot(fig2)

    st.write("**WordCloud From Both Documents**")
    st.pyplot(fig3)

    st.write("**Bi-Gram for Document 1**")
    st.pyplot(fig4)

    st.write("**Bi-Gram for Document 2**")
    st.pyplot(fig5)


    def to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
    
    # Configure genai with API key
    genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

    # Instantiate the model
    model = genai.GenerativeModel('gemini-1.0-pro-latest')

    # Generate content
    response = model.generate_content(["Compare the simmilarities and give some conclusion between these 2 PDF Document : ", hasil1, "and", hasil2], stream=True)
    response.resolve()
    st.write("**Google Gemini Response About Data**")
    st.write(response.text)