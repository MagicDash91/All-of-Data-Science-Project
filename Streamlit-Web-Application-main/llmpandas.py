import pandas as pd
import streamlit as st
from langchain_groq.chat_models import ChatGroq
from pandasai import SmartDataframe
import os
from PIL import Image
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import io
import matplotlib.pyplot as plt

# Load language model
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    api_key="YOUR_GROQ_API")

def main():
    st.title("Ask your CSV")

    # Allow user to upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read uploaded CSV file into pandas DataFrame
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)

        # Convert DataFrame into SmartDataFrame
        df = SmartDataframe(data, config={"llm": llm})

        # Add text box for user input
        question = st.text_input("Ask a question about the data:")

        if st.button("Ask"):
            if question:
                # Answer the user's question using the language model
                answer = df.chat(question)

                # Display the answer
                st.write("Answer:", answer)

                # Check if the answer is a visualization
                if isinstance(answer, str) and os.path.exists(answer):
                    # Open the image file
                    image = Image.open(answer)
                    # Display the image
                    st.image(image, caption="Visualization")

                    # Save the figure as result.png
                    plt.savefig("result.png")

                    # Generate content using Google Gemini
                    def to_markdown(text):
                        text = text.replace('•', '  *')
                        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

                    genai.configure(api_key="YOUR_GOOGLE_GEMINI_API")
                    model = genai.GenerativeModel('gemini-pro-vision')

                    img1 = Image.open("result.png")
                    response = model.generate_content(["You are a Professional Data Analyst, give a conclusion and actionable insight based on the visualization", img1], stream=True)
                    response.resolve()

                    st.write("**Google Gemini Response About Data**")
                    st.write(response.text)
                else:
                    st.warning("No visualization found.")
            else:
                st.warning("Please ask a question.")

if __name__ == "__main__":
    main()






