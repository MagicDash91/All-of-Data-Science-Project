import streamlit as st
import requests
from bs4 import BeautifulSoup
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True)

st.title("Web Article Summarizer")

target_url = st.text_input("Enter the target URL:")
process_button = st.button("Scrape Text")  # Button text adjusted

def scrape_text(url):
  """Scrapes text from a website and returns the extracted text.

  Args:
      url: The URL of the website to scrape.

  Returns:
      The scraped text content as a string, or None if there's an error.
  """

  if not url:  # Check if URL is empty
    return None

  try:
    # Send HTTP request and parse HTML content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract text based on your desired method (modify as needed)
    # Here, we're extracting text from all paragraphs
    paragraphs = soup.find_all("p")
    paragraph_text = []
    for paragraph in paragraphs[:2]:  # Limit to first 2 paragraphs
      paragraph_text.append(paragraph.text.strip())

    # Combine text from all paragraphs (limited to first 2)
    all_paragraph_text = "\n".join(paragraph_text)

    return all_paragraph_text
  except Exception as e:
    st.error(f"Error scraping text: {e}")
    return None

if process_button:  # Only execute if button is clicked
  scraped_text = scrape_text(target_url)

  if scraped_text:
    st.success("Text scraped successfully!")
    st.subheader("Showing First Paragraphs of Article:")
    st.write(scraped_text)  # Show only the first 2 paragraphs

    # Load English tokenizer, tagger, parser and NER
    nlp = spacy.load("en_core_web_sm")

    # Process the scraped text
    doc = nlp(scraped_text)

    # Analyze syntax - Extract Noun Phrases
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    # Create DataFrame using Pandas (alternative to columns argument)
    noun_phrases_df = pd.DataFrame(noun_phrases, columns=["Noun Phrase"])  # Create DataFrame with Pandas

    # Display Noun Phrases in Streamlit table
    st.subheader("Noun Phrases:")
    st.dataframe(noun_phrases_df)

    # Analyze syntax - Extract Verbs
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

    # Create DataFrame for Verbs
    verbs_df = pd.DataFrame(verbs, columns=["Verb"])

    # Display Verbs in Streamlit table
    st.subheader("Verbs:")
    st.dataframe(verbs_df)


    # Analyze Part-of-Speech Distribution
    pos_counts = {token.pos_: 0 for token in doc}
    for token in doc:
      pos_counts[token.pos_] += 1

    # Create Part-of-Speech Distribution Plot (using matplotlib)
    plt.figure(figsize=(8, 6))
    plt.bar(pos_counts.keys(), pos_counts.values())
    plt.xlabel("Part of Speech")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display Part-of-Speech Distribution Plot in Streamlit
    st.subheader("Part-of-Speech Distribution :")
    st.pyplot(plt)

  else:
    st.warning("No text found on the provided URL or an error occurred.")








