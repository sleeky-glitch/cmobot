import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
import re
import os

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Newspaper Bot for CMO by BSPL",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto"
)

# Initialize OpenAI API key
openai.api_key = st.secrets.get("openai_key", None)
if not openai.api_key:
    st.error("OpenAI API key is missing. Please add it to the Streamlit secrets.")
    st.stop()

st.title("Newspaper Bot for the CMO")

# Initialize session state for messages and references
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome! I can help you find news related to events happening in the state of Gujarat. Please ask your question!"
        }
    ]

if "references" not in st.session_state:
    st.session_state.references = []

@st.cache_resource(show_spinner=False)
def load_data():
    try:
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=2048,
            chunk_overlap=100,
        )

        reader = SimpleDirectoryReader(
            input_dir="./news_articles",  # Updated to use the correct directory name
            recursive=True,
            filename_as_id=True
        )
        docs = reader.load_data()

        Settings.llm = OpenAI(
            model="gpt-4",
            temperature=0.1,
            system_prompt="""You are an authoritative expert on Gujarati News, all your answers have to be in Gujarati.
            Your responses should be:
            1. Comprehensive and detailed
            2. Give complete news article whenever possible
            3. Quote relevant sections directly from the Given Data
            4. Provide specific references Title and Date
            5. Break down complex news into numbered steps
            6. Include any relevant timelines or deadlines
            7. Highlight important caveats or exceptions"""
        )

        index = VectorStoreIndex.from_documents(
            docs,
            node_parser=node_parser,
            show_progress=True
        )
        return index
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def extract_references(text):
    pattern = r'Source: ([^,]+), Page (\d+)'  # Removed $ anchors
    matches = re.finditer(pattern, text)
    references = []

    for match in matches:
        doc_name, page = match.groups()
        link = f'<a href="news_articles/{doc_name}.pdf#page={page}" target="_blank">[Source: {doc_name}, Page {page}]</a>'
        text = text.replace(match.group(0), link)
        references.append((doc_name, page))

    st.session_state.references = list(set(references))
    return text

def format_response(response):
    formatted_response = extract_references(response)
    formatted_response = formatted_response.replace("Step ", "\n### Step ")
    formatted_response = formatted_response.replace("Note:", "\n> **Note:**")
    formatted_response = formatted_response.replace("Important:", "\n> **Important:**")
    return formatted_response

def list_reference_documents():
    try:
        files = os.listdir('./news_articles')  # Updated to use the correct directory name
        text_files = [f for f in files if f.endswith('.txt') or f.endswith('.pdf')]  # Added PDF files
        if text_files:
            for file in text_files:
                doc_name = os.path.splitext(file)[0]
                st.markdown(f'- [{doc_name}](./news_articles/{file})', unsafe_allow_html=True)  # Fixed link formatting
        else:
            st.write("No reference documents found.")
    except Exception as e:
        st.error(f"Error listing documents: {e}")

# Load the index
index = load_data()

# Initialize chat engine
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        verbose=True
    )

# Sidebar for reference documents
with st.sidebar:
    st.header("📚 Reference Documents")
    st.write("Available reference documents:")
    list_reference_documents()

    st.header("🔗 References Used")
    if st.session_state.references:
        for doc_name, page in st.session_state.references:
            st.markdown(f'- [Source: {doc_name}, Page {page}](./news_articles/{doc_name}.pdf#page={page})', unsafe_allow_html=True)
    else:
        st.write("No references used yet.")

# Chat interface
if prompt := st.chat_input("Ask a question about News in Gujarat"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Generate new response
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        try:
            response = st.session_state.chat_engine.chat(prompt)
            formatted_response = format_response(response.response)

            st.markdown(formatted_response, unsafe_allow_html=True)

            message = {
                "role": "assistant",
                "content": formatted_response
            }
            st.session_state.messages.append(message)
        except Exception as e:
            st.error(f"Error generating response: {e}")

# Add CSS for better formatting
st.markdown("""
<style>
a {
    color: #0078ff;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)
