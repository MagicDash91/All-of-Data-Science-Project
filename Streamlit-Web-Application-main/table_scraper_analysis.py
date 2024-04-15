import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown

def scrape_tables(url):
  """
  Scrapes all tables from a given URL and returns them as a list of DataFrames.

  Args:
      url: The URL of the webpage to scrape.

  Returns:
      A list of pandas DataFrames, each representing a scraped table.
  """
  # Fetch the HTML content
  response = requests.get(url)
  response.raise_for_status()  # Raise an error if the request fails

  # Parse the HTML content
  soup = BeautifulSoup(response.content, "html.parser")

  # Find all tables
  tables = soup.find_all("table")

  # Extract data and convert to DataFrames
  all_dataframes = []
  for table in tables:
    # Extract rows from the table
    rows = table.find_all("tr")
    table_data = []
    for row in rows:
      # Extract cells from each row
      cells = row.find_all(["th", "td"])  # Consider both headers and data cells
      row_data = [cell.text.strip() for cell in cells]  # Extract text and strip whitespace
      table_data.append(row_data)

    # Check if there's data before creating a DataFrame
    if table_data:
      df = pd.DataFrame(table_data)
      all_dataframes.append(df)

  return all_dataframes

def display_and_modify_tables(dataframes):
  """
  Displays scraped DataFrames in Streamlit and allows user interaction for modifications.

  Args:
      dataframes: A list of pandas DataFrames containing scraped data.
  """
  # Display all scraped tables (head)
  if dataframes:
    st.subheader("Scraped Tables:")
    for i, df in enumerate(dataframes):
      st.write(f"Table {i+1}")
      st.dataframe(df.head())  # Show only the head (first few rows)

    # Table selection for modification
    selected_table_index = st.selectbox("Select a Table to Modify", range(len(dataframes)))
    selected_df = dataframes[selected_table_index]

    # Display the full selected table
    st.subheader(f"Selected Table {selected_table_index+1}")
    st.dataframe(selected_df)

    # Row selection for removal with multi-select
    rows_to_remove = st.multiselect("Select rows to remove (0-based):", selected_df.index.tolist(), key="rows_to_remove")

    # Combined button for row removal with confirmation
    if st.button("Remove Selected Rows"):
      if rows_to_remove:  # Check if any rows were selected
        try:
          selected_df.drop(rows_to_remove, axis=0, inplace=True)  # Remove rows
          st.success(f"Selected rows removed successfully!")
          # Display the modified DataFrame
          st.subheader(f"Modified Table {selected_table_index+1}")
          st.dataframe(selected_df)
        except Exception as e:
          st.error(f"Error removing rows: {e}")

    # --- Google Gemini Integration ---
    # Convert the DataFrame to a string variable
    df_string = selected_df.to_string()

    # Configure genai with API key (replace with your actual key)
    genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")  # Replace with your Google GenerativeAI API key

    model = genai.GenerativeModel('gemini-1.0-pro-latest')

    try:
        # Generate content with Gemini
        response = model.generate_content(["You are a Professional Data Analyst, Make a Summary and actionable insight based on the csv dataset here :", df_string], stream=True)
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)
    except Exception as e:
        st.error(f"Error generating content with Google Gemini: {e}")

  
# Streamlit app
st.title("Table Scraper and Modifier App")
url = st.text_input("Enter the URL to scrape:")
if url:
    try:
        scraped_dataframes = scrape_tables(url)
        display_and_modify_tables(scraped_dataframes)
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred scraping the URL: {e}")
        


