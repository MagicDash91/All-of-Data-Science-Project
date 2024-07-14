import asyncio
import os
import streamlit as st
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

# Title of the app
st.title("PDF Document Comparer Analysis")

# Upload the PDF files
uploaded_file1 = st.file_uploader("Upload First PDF file:", type='pdf')
uploaded_file2 = st.file_uploader("Upload Second PDF file:", type='pdf')
question = st.text_input("Insert Question", "Put your question here about both documents")

async def process_files():
    if uploaded_file1 and uploaded_file2 and question:
        # Save the uploaded files as file1.pdf and file2.pdf
        file1_path = "file1.pdf"
        file2_path = "file2.pdf"
        with open(file1_path, "wb") as f1:
            f1.write(uploaded_file1.getbuffer())
        with open(file2_path, "wb") as f2:
            f2.write(uploaded_file2.getbuffer())

        # Initialize the LLM with the Google API key
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

        # Load the PDF files
        loader1 = PyPDFLoader(file1_path)
        loader2 = PyPDFLoader(file2_path)
        docs1 = loader1.load()
        docs2 = loader2.load()
        docs3 = docs1 + docs2

        # Define the Summarize Chain
        template = """Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

        # Process both documents
        response1 = stuff_chain.invoke(docs1)
        response2 = stuff_chain.invoke(docs2)

        # Display the summaries
        st.markdown("### Summary of the First Document")
        st.write(response1["output_text"])

        st.markdown("### Summary of the Second Document")
        st.write(response2["output_text"])

        # Additional comparison logic can be added here based on the question
        comparison_template = question + """Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:"""

        prompt1 = PromptTemplate.from_template(comparison_template)
        llm_chain1 = LLMChain(llm=llm, prompt=prompt1)
        stuff_chain1 = StuffDocumentsChain(llm_chain=llm_chain1, document_variable_name="text")
        response3 = stuff_chain1.invoke(docs3)

        # Display the comparison result
        st.markdown("### Comparison Result")
        st.write(response3["output_text"])

        # Clean up the temporary files
        os.remove(uploaded_file1.name)
        os.remove(uploaded_file2.name)

if st.button("Process"):
    asyncio.run(process_files())
