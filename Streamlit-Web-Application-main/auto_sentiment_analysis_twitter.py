import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(color_codes=True)
import os
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import PIL.Image

st.title("Sentiment Analysis")

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud
import PyPDF2
import re
from io import StringIO
import plotly.express as px
import pandas as pd
import collections
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Create stopword remover
stop_factory = StopWordRemoverFactory()
more_stopword = ['dengan', 'ia', 'bahwa', 'oleh', 'rp', 'undang', 'pasal', 'ayat', 'bab']
data = stop_factory.get_stop_words() + more_stopword

# Upload the CSV file
uploaded_file = st.file_uploader("Upload CSV file:")

# User input for delimiter
delimiter_option = st.radio("Select CSV delimiter:", [",", ";"], index=0)

# Add custom stopwords
custom_stopwords = st.text_input("Enter custom stopwords (comma-separated):")
custom_stopword_list = [word.strip() for word in custom_stopwords.split(",")] if custom_stopwords else []

# Check if the file is uploaded
if uploaded_file is not None:
    # Read the CSV file into a Pandas DataFrame
    if delimiter_option == ",":
        df = pd.read_csv(uploaded_file, delimiter=",")
    elif delimiter_option == ";":
        df = pd.read_csv(uploaded_file, delimiter=";")
    else:
        st.error("Invalid delimiter option.")

    # Show the DataFrame
    st.dataframe(df)

    # Select a column for sentiment analysis
    object_columns = df.select_dtypes(include="object").columns
    target_variable = st.selectbox("Choose a column for Sentiment Analysis:", object_columns)

    # Perform sentiment analysis on the selected column
    if st.button("Perform Sentiment Analysis"):
        # Your sentiment analysis logic goes here
        st.success(f"Sentiment Analysis performed on column: {target_variable}")
        
        # Show the selected column
        st.write(f"Selected {target_variable} Column:")
        st.dataframe(df[[target_variable]])

        # Create a new DataFrame with cleaned text column
        new_df = df.copy()

        # Create cleaned text column (updated to include custom stopwords)
        custom_stopword_list = [word.strip() for word in custom_stopwords.split(",")] if custom_stopwords else []
        new_df['cleaned_text'] = new_df[target_variable].apply(lambda x: ' '.join(
            [stemmer.stem(word) for word in stop_factory.create_stop_word_remover().remove(x).split() 
            if word.lower() not in data and word.lower() not in custom_stopword_list]  # Exclude custom stopwords
        ))

        # Apply stemming and stopword removal to the selected column
        new_df['cleaned_text'] = new_df[target_variable].apply(lambda x: ' '.join([stemmer.stem(word) for word in stop_factory.create_stop_word_remover().remove(x).split() if word.lower() not in data]))

        # Show the cleaned text column
        #st.write("Cleaned Text Column:")
        #st.dataframe(new_df[['cleaned_text']])

        # Load the sentiment analysis pipeline
        pretrained = "indonesia-bert-sentiment-classification"
        model = AutoModelForSequenceClassification.from_pretrained(pretrained)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

        # Function to apply sentiment analysis to each row in the 'cleaned_text' column
        def analyze_sentiment(text):
            result = sentiment_analysis(text)
            label = label_index[result[0]['label']]
            score = result[0]['score']
            return pd.Series({'sentiment_label': label, 'sentiment_score': score})

        # Apply sentiment analysis to 'cleaned_text' column
        new_df[['sentiment_label', 'sentiment_score']] = new_df['cleaned_text'].apply(analyze_sentiment)

        # Display the results
        st.write("Sentiment Analysis Results:")
        st.dataframe(new_df[['cleaned_text', 'sentiment_label', 'sentiment_score']])

        # Count the occurrences of each sentiment label
        sentiment_counts = new_df['sentiment_label'].value_counts()

        # Plot a bar chart using seaborn
        st.set_option('deprecation.showPyplotGlobalUse', False)
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Count')
        st.pyplot()

        # Define a dictionary to store sentiment-wise text
        sentiment_text = {
            "positive": "",
            "neutral": "",
            "negative": ""
        }

        # Loop through each sentiment label
        for label in sentiment_counts.index:
            # Filter data for the current sentiment
            selected_data = new_df[new_df['sentiment_label'] == label]

            # Include custom stopwords back into the cleaned text before concatenation
            selected_data['cleaned_text'] = selected_data['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in data and word.lower() not in custom_stopword_list]))  # Remove only general stopwords

            # Concatenate cleaned text from the selected data (now including custom stopwords)
            sentiment_text[label] = ' '.join(selected_data['cleaned_text'].astype(str))


        # Define variables for sentiment-wise text (adjust variable names)
        #positive_text = ""
        #neutral_text = ""
        #negative_text = ""


        # Concatenate cleaned text for each sentiment
        positive_text = ' '.join([word for word in new_df[new_df['sentiment_label'] == 'positive']['cleaned_text'].apply(lambda x: ' '.join([w for w in x.split() if w.lower() not in data and w.lower() not in custom_stopword_list]))])
        neutral_text = ' '.join([word for word in new_df[new_df['sentiment_label'] == 'neutral']['cleaned_text'].apply(lambda x: ' '.join([w for w in x.split() if w.lower() not in data and w.lower() not in custom_stopword_list]))])
        negative_text = ' '.join([word for word in new_df[new_df['sentiment_label'] == 'negative']['cleaned_text'].apply(lambda x: ' '.join([w for w in x.split() if w.lower() not in data and w.lower() not in custom_stopword_list]))])


       
        # Generate WordCloud for positive sentiment
        positive_wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='viridis', background_color='white'
        ).generate(positive_text)

        # Save the WordCloud image with a filename
        positive_wordcloud_filename = "wordcloud_positive.png"
        positive_wordcloud.to_file(positive_wordcloud_filename)

        # Display the saved WordCloud image using Streamlit
        st.subheader("WordCloud for Positive Sentiment")
        st.image(positive_wordcloud_filename)

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

        import PIL.Image

        img = PIL.Image.open("wordcloud_positive.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image about wordcloud positive sentiment", img])
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)


        # Generate WordCloud for neutral sentiment
        neutral_wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='viridis', background_color='white'
        ).generate(neutral_text)

        # Save the WordCloud image with a filename
        neutral_wordcloud_filename = "wordcloud_neutral.png"
        neutral_wordcloud.to_file(neutral_wordcloud_filename)

        # Display the saved WordCloud image using Streamlit
        st.subheader("WordCloud for Neutral Sentiment")
        st.image(neutral_wordcloud_filename)

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

        import PIL.Image

        img = PIL.Image.open("wordcloud_neutral.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image about wordcloud neutral sentiment", img])
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)



        # Generate WordCloud for negative sentiment
        negative_wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='viridis', background_color='white'
        ).generate(negative_text)

        # Save the WordCloud image with a filename
        negative_wordcloud_filename = "wordcloud_negative.png"
        negative_wordcloud.to_file(negative_wordcloud_filename)

        # Display the saved WordCloud image using Streamlit
        st.subheader("WordCloud for Negative Sentiment")
        st.image(negative_wordcloud_filename)

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

        import PIL.Image

        img = PIL.Image.open("wordcloud_negative.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image about wordcloud negative sentiment", img])
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)


        # Bigrams Positive Sentiment
        words1 = positive_text.split()
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
        plt.title(f"Top 10 Bigram for Positive Sentiment")
        # Save the entire plot as a PNG
        plt.tight_layout()
        plt.savefig("bigram_positive.png")
        st.subheader("Bigram for Positive Sentiment")
        st.image("bigram_positive.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

        import PIL.Image

        img = PIL.Image.open("bigram_positive.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image about bigram positive sentiment", img])
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

        genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

        import PIL.Image

        img = PIL.Image.open("bigram_neutral.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image about bigram neutral sentiment", img])
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)



        # Bigrams Negative Sentiment
        words1 = negative_text.split()
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
        plt.title(f"Top 10 Bigram for negative Sentiment")
        # Save the entire plot as a PNG
        plt.tight_layout()
        plt.savefig("bigram_negative.png")
        st.subheader("Bigram for Negative Sentiment")
        st.image("bigram_negative.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

        import PIL.Image

        img = PIL.Image.open("bigram_negative.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image about bigram negative sentiment", img])
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)
