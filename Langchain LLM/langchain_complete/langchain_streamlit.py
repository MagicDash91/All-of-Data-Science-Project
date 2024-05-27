import streamlit as st

# Define functions for each page
def langchain_pdf():
    st.title("Langchain PDF Text Analysis")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.chains import StuffDocumentsChain
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    # Initialize Model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyAQLXJ6ROBzMycImPVp2jTlbB3zIpEWmhM")

    # Input for PDF file
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        
    # Input for the question
    question = st.text_input("Enter your question:")

    if st.button("Analyze"):
        if uploaded_file is not None:
            # Save the uploaded PDF file with the name "file.pdf"
            with open("file.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load the PDF file
            loader = PyPDFLoader("file.pdf")
            docs = loader.load_and_split()

            # Define the Summarize Chain
            template = question + """ Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:"""

            prompt = PromptTemplate.from_template(template)

            llm_chain = LLMChain(llm=llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

            # Invoke Chain
            response = stuff_chain.invoke(docs)
            summary = response["output_text"]

            # Display the summary
            st.header("Summary:")
            st.write(summary)
        else:
            st.error("Please upload a PDF file.")


def langchain_doc():
    st.title("Langchain Microsoft Word File Analysis")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.document_loaders import Docx2txtLoader
    from langchain.chains import StuffDocumentsChain
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    # Initialize Model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyAQLXJ6ROBzMycImPVp2jTlbB3zIpEWmhM")

    # Input for PDF file
    uploaded_file = st.file_uploader("Upload PDF", type=["docx"])
        
    # Input for the question
    question = st.text_input("Enter your question:")

    if st.button("Analyze"):
        if uploaded_file is not None:
            # Save the uploaded PDF file with the name "file.pdf"
            with open("file.docx", "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load the PDF file
            loader = Docx2txtLoader("file.docx")
            docs = loader.load_and_split()

            # Define the Summarize Chain
            template = question + """ Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:"""

            prompt = PromptTemplate.from_template(template)

            llm_chain = LLMChain(llm=llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

            # Invoke Chain
            response = stuff_chain.invoke(docs)
            summary = response["output_text"]

            # Display the summary
            st.header("Summary:")
            st.write(summary)
        else:
            st.error("Please upload a Micosoft Word file.")


def langchain_excel():
    st.title("Langchain Microsoft Excel File Analysis")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.document_loaders import UnstructuredExcelLoader
    from langchain.chains import StuffDocumentsChain
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    # Initialize Model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyAQLXJ6ROBzMycImPVp2jTlbB3zIpEWmhM")

    # Input for PDF file
    uploaded_file = st.file_uploader("Upload PDF", type=["xlsx"])
        
    # Input for the question
    question = st.text_input("Enter your question:")

    if st.button("Analyze"):
        if uploaded_file is not None:
            # Save the uploaded PDF file with the name "file.pdf"
            with open("file.xlsx", "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load the PDF file
            loader = UnstructuredExcelLoader("file.xlsx", mode="elements")
            docs = loader.load()

            # Define the Summarize Chain
            template = question + """ Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:"""

            prompt = PromptTemplate.from_template(template)

            llm_chain = LLMChain(llm=llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

            # Invoke Chain
            response = stuff_chain.invoke(docs)
            summary = response["output_text"]

            # Display the summary
            st.header("Summary:")
            st.write(summary)
        else:
            st.error("Please upload a Excel file.")

def langchain_ppt():
    st.title("Langchain Microsoft Power Point File Analysis")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
    from langchain.chains import StuffDocumentsChain
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    # Initialize Model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyAQLXJ6ROBzMycImPVp2jTlbB3zIpEWmhM")

    # Input for PDF file
    uploaded_file = st.file_uploader("Upload PDF", type=["pptx"])
        
    # Input for the question
    question = st.text_input("Enter your question:")

    if st.button("Analyze"):
        if uploaded_file is not None:
            # Save the uploaded PDF file with the name "file.pdf"
            with open("file.pptx", "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load the PDF file
            loader = UnstructuredPowerPointLoader("file.pptx", mode="elements")
            docs = loader.load_and_split()

            # Define the Summarize Chain
            template = question + """ Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:"""

            prompt = PromptTemplate.from_template(template)

            llm_chain = LLMChain(llm=llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

            # Invoke Chain
            response = stuff_chain.invoke(docs)
            summary = response["output_text"]

            # Display the summary
            st.header("Summary:")
            st.write(summary)
        else:
            st.error("Please upload a Excel file.")

def langchain_csv():
    st.title("Langchain CSV File Analysis")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.document_loaders.csv_loader import CSVLoader
    from langchain.chains import StuffDocumentsChain
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    # Initialize Model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyAQLXJ6ROBzMycImPVp2jTlbB3zIpEWmhM")

    # Input for PDF file
    uploaded_file = st.file_uploader("Upload PDF", type=["csv"])
        
    # Input for the question
    question = st.text_input("Enter your question:")

    if st.button("Analyze"):
        if uploaded_file is not None:
            # Save the uploaded PDF file with the name "file.pdf"
            with open("file.csv", "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load the PDF file
            loader = CSVLoader(file_path="file.csv")
            docs = loader.load()

            # Define the Summarize Chain
            template = question + """ Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:"""

            prompt = PromptTemplate.from_template(template)

            llm_chain = LLMChain(llm=llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

            # Invoke Chain
            response = stuff_chain.invoke(docs)
            summary = response["output_text"]

            # Display the summary
            st.header("Summary:")
            st.write(summary)
        else:
            st.error("Please upload a CSV file.")

def langchain_web():
    st.title("Langchain Web Content Analysis")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.document_loaders import WebBaseLoader
    from langchain.chains import StuffDocumentsChain
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    # Initialize Model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyAQLXJ6ROBzMycImPVp2jTlbB3zIpEWmhM")

    # Input for article link
    article_link = st.text_input("Enter the link to the article:")
        
    # Input for the question
    question = st.text_input("Enter your question:")

    if st.button("Analyze"):
        if article_link.strip() == "":
            st.error("Please enter a link to the article.")
        else:
            # Load the article content
            loader = WebBaseLoader(article_link)
            docs = loader.load()

            # Define the Summarize Chain
            template = question + """ Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:"""

            prompt = PromptTemplate.from_template(template)

            llm_chain = LLMChain(llm=llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

            # Invoke Chain
            response = stuff_chain.invoke(docs)
            summary = response["output_text"]

            # Display the summary
            st.header("Summary:")
            st.write(summary)

def langchain_youtube():
    st.title("Langchain Youtube Video Analysis")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.document_loaders import YoutubeLoader
    from langchain.chains import StuffDocumentsChain
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    # Initialize Model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyAQLXJ6ROBzMycImPVp2jTlbB3zIpEWmhM")

    # Input for article link
    youtube_link = st.text_input("Enter the YouTube link:")
        
    # Input for the question
    question = st.text_input("Enter your question:")

    if st.button("Analyze"):
        if youtube_link.strip() == "":
            st.error("Please enter a link to the article.")
        else:
            # Load the article content
            loader = YoutubeLoader.from_youtube_url(
                youtube_link,
                add_video_info=True,
                language=["en", "id"],
                translation="en",
            )
            docs = loader.load()

            # Define the Summarize Chain
            template = question + """ Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:"""

            prompt = PromptTemplate.from_template(template)

            llm_chain = LLMChain(llm=llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

            # Invoke Chain
            response = stuff_chain.invoke(docs)
            summary = response["output_text"]

            # Display the summary
            st.header("Summary:")
            st.write(summary)

# Set CSS to arrange buttons horizontally
st.markdown(
    """
    <style>
        .sidebar .widget-button {
            width: 100%;
            white-space: normal;
            text-align: left;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Get the selected page
selected_page = st.sidebar.radio(
    "Select Page",
    ("Langchain PDF Text Analysis", 
     "Langchain Microsoft Word File Analysis", 
     "Langchain Microsoft Excel File Analysis", 
     "Langchain Microsoft Power Point File Analysis",
     "Langchain CSV File Analysis", 
     "Langchain Web Content Analysis",
     "Langchain Youtube Video Analysis")
)

if selected_page == "Langchain PDF Text Analysis":
    langchain_pdf()
elif selected_page == "Langchain Microsoft Word File Analysis":
    langchain_doc()
elif selected_page == "Langchain Microsoft Excel File Analysis":
    langchain_excel()
elif selected_page == "Langchain Microsoft Power Point File Analysis":
    langchain_ppt()
elif selected_page == "Langchain CSV File Analysis":
    langchain_csv()
elif selected_page == "Langchain Web Content Analysis":
    langchain_web()
elif selected_page == "Langchain Youtube Video Analysis":
    langchain_youtube()
