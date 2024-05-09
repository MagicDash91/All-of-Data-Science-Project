import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import collections
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown

# Create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Create stopword remover
stop_factory = StopWordRemoverFactory()
more_stopword = ['dengan', 'ia', 'bahwa', 'oleh', 'rp', 'undang', 'pasal', 'ayat', 'bab']
data = stop_factory.get_stop_words() + more_stopword

# Define patterns for removal
hyperlink_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
number_pattern = re.compile(r'\b\d+\b')
emoticon_pattern = re.compile(u'('
    u'\ud83c[\udf00-\udfff]|'
    u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
    u'[\u2600-\u26FF\u2700-\u27BF])+', 
    re.UNICODE)

st.title('Sentiment Analysis')

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
custom_stopwords = st.text_input('Custom Stopwords (comma-separated)', '')

if uploaded_file is not None and custom_stopwords:
    if st.button('Analyze'):
        df = pd.read_csv(uploaded_file)
        custom_stopword_list = [word.strip() for word in custom_stopwords.split(',')]
        all_stopwords = data + custom_stopword_list

        df['cleaned_text'] = df['full_text'].str.replace(hyperlink_pattern, '')
        df['cleaned_text'] = df['cleaned_text'].str.replace(emoticon_pattern, '')
        df['cleaned_text'] = df['cleaned_text'].str.replace(number_pattern, '')

        for stopword in custom_stopword_list:
            df['cleaned_text'] = df['cleaned_text'].str.replace(stopword, '')

        df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(
            [stemmer.stem(word) for word in stop_factory.create_stop_word_remover().remove(x).split()
            if word.lower() not in all_stopwords]
        ))

        from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

        tokenizer = BertTokenizer.from_pretrained("indobert-emotion-classification")
        config = BertConfig.from_pretrained("indobert-emotion-classification")
        model = BertForSequenceClassification.from_pretrained("indobert-emotion-classification", config=config)
        from transformers import pipeline

        nlp = pipeline("text-classification", model="indobert-emotion-classification")
        results = df['cleaned_text'].apply(lambda x: nlp(x)[0])
        df['label'] = [res['label'] for res in results]
        df['score'] = [res['score'] for res in results]

        sentiment_counts = df['label'].value_counts()

        st.write("### Sentiment Distribution")
        st.bar_chart(sentiment_counts)

        st.write("### Analysis Results")
        st.write(df)

        anger_text = ' '.join(df[df['label'] == 'Anger']['cleaned_text'])
        happy_text = ' '.join(df[df['label'] == 'Happy']['cleaned_text'])
        neutral_text = ' '.join(df[df['label'] == 'Neutral']['cleaned_text'])
        fear_text = ' '.join(df[df['label'] == 'Fear']['cleaned_text'])
        sadness_text = ' '.join(df[df['label'] == 'Sadness']['cleaned_text'])
        love_text = ' '.join(df[df['label'] == 'Love']['cleaned_text'])

        # Bigrams Anger Sentiment
        words1 = anger_text.split()
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
        plt.title(f"Top 10 Bigram for Anger Sentiment")
        # Save the entire plot as a PNG
        plt.tight_layout()
        plt.savefig("bigram_anger.png")
        st.subheader("Bigram for Anger Sentiment")
        st.image("bigram_anger.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")

        import PIL.Image

        img = PIL.Image.open("bigram_anger.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["As a marketing consulant, I want to understand consumer insighst based on the chart and the market context so I can use the key findings to formulate actionable insights", img])
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)




        # Bigrams Happy Sentiment
        words1 = happy_text.split()
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
        plt.title(f"Top 10 Bigram for Happy Sentiment")
        # Save the entire plot as a PNG
        plt.tight_layout()
        plt.savefig("bigram_happy.png")
        st.subheader("Bigram for Happy Sentiment")
        st.image("bigram_happy.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")

        import PIL.Image

        img = PIL.Image.open("bigram_happy.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["As a marketing consulant, I want to understand consumer insighst based on the chart and the market context so I can use the key findings to formulate actionable insights", img])
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)




        # Bigrams Neutral Sentiment
        words1 = neutral_text.split()
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
        plt.title(f"Top 10 Bigram for Neutral Sentiment")
        # Save the entire plot as a PNG
        plt.tight_layout()
        plt.savefig("bigram_neutral.png")
        st.subheader("Bigram for Neutral Sentiment")
        st.image("bigram_neutral.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")

        import PIL.Image

        img = PIL.Image.open("bigram_neutral.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["As a marketing consulant, I want to understand consumer insighst based on the chart and the market context so I can use the key findings to formulate actionable insights", img])
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)




        # Bigrams Fear Sentiment
        words1 = fear_text.split()
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
        plt.title(f"Top 10 Bigram for Fear Sentiment")
        # Save the entire plot as a PNG
        plt.tight_layout()
        plt.savefig("bigram_fear.png")
        st.subheader("Bigram for Fear Sentiment")
        st.image("bigram_fear.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")

        import PIL.Image

        img = PIL.Image.open("bigram_fear.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["As a marketing consulant, I want to understand consumer insighst based on the chart and the market context so I can use the key findings to formulate actionable insights", img])
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)




        # Bigrams Sadness Sentiment
        words1 = sadness_text.split()
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
        plt.title(f"Top 10 Bigram for Sadness Sentiment")
        # Save the entire plot as a PNG
        plt.tight_layout()
        plt.savefig("bigram_sadness.png")
        st.subheader("Bigram for Sadness Sentiment")
        st.image("bigram_sadness.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")

        import PIL.Image

        img = PIL.Image.open("bigram_sadness.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["As a marketing consulant, I want to understand consumer insighst based on the chart and the market context so I can use the key findings to formulate actionable insights", img])
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)




        # Bigrams Love Sentiment
        words1 = love_text.split()
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
        plt.title(f"Top 10 Bigram for Love Sentiment")
        # Save the entire plot as a PNG
        plt.tight_layout()
        plt.savefig("bigram_love.png")
        st.subheader("Bigram for Love Sentiment")
        st.image("bigram_love.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")

        import PIL.Image

        img = PIL.Image.open("bigram_love.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["As a marketing consulant, I want to understand consumer insighst based on the chart and the market context so I can use the key findings to formulate actionable insights", img])
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)