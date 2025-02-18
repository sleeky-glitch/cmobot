import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from typing import List, Dict, Any, Optional
import logging
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_language(text: str) -> str:
    """
    Detect if text contains Gujarati characters
    """
    gujarati_range = range(0x0A80, 0x0AFF)
    for char in text:
        if ord(char) in gujarati_range:
            return 'gu'
    return 'en'

def translate_to_gujarati(text: str) -> str:
    """
    Translate English text to Gujarati using deep_translator
    """
    try:
        # Detect language
        detected_lang = detect_language(text)
        if detected_lang == 'gu':
            return text

        # Initialize translator
        translator = GoogleTranslator(source='en', target='gu')

        # Translate text
        translation = translator.translate(text)
        logger.info(f"Translated '{text}' to '{translation}'")
        return translation
    except Exception as e:
        logger.error(f"Error in translation: {e}")
        return text

# Initialize OpenAI and Pinecone clients
@st.cache_resource
def init_clients():
    try:
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        return openai_client, pc
    except Exception as e:
        logger.error(f"Error initializing clients: {e}")
        st.error(f"Error initializing clients: {e}")
        st.stop()

# Initialize the clients
openai_client, pc = init_clients()

# Get Pinecone index
try:
    index_name = "gujarati-news"
    index = pc.Index(index_name)
except Exception as e:
    logger.error(f"Error connecting to Pinecone index: {e}")
    st.error(f"Error connecting to Pinecone index: {e}")
    st.stop()

def is_valid_embedding(embedding: Optional[List[float]]) -> bool:
    """
    Check if an embedding is valid
    """
    if not embedding:
        return False
    try:
        embedding_array = np.array(embedding)
        return embedding_array.size > 0 and not np.isnan(embedding_array).any()
    except Exception as e:
        logger.error(f"Error validating embedding: {e}")
        return False

def safe_get_embedding(text: str) -> Optional[List[float]]:
    """
    Safely get embedding from OpenAI
    """
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        if is_valid_embedding(embedding):
            return embedding
        return None
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        st.error(f"Error getting embedding: {e}")
        return None

def calculate_relevance_score(query_embedding: List[float],
                            result_embedding: List[float],
                            content: str) -> float:
    """
    Calculate a combined relevance score based on embedding similarity
    """
    try:
        query_embedding = np.array(query_embedding).reshape(1, -1)
        result_embedding = np.array(result_embedding).reshape(1, -1)
        similarity = cosine_similarity(query_embedding, result_embedding)[0][0]
        normalized_score = (similarity + 1) / 2
        return normalized_score
    except Exception as e:
        logger.error(f"Error in calculate_relevance_score: {e}")
        return 0.0

def retry_with_backoff(func, max_retries=3, initial_delay=1):
    """
    Retry a function with exponential backoff
    """
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if i == max_retries - 1:
                raise e
            time.sleep(initial_delay * (2 ** i))

def filter_relevant_results(results: List[Any],
                          query_embedding: List[float],
                          min_similarity: float = 0.6) -> List[Any]:
    """
    Filter results based on relevance criteria
    """
    filtered_results = []
    for match in results:
        try:
            result_embedding = match.values
            if result_embedding and len(result_embedding) > 0:
                relevance_score = calculate_relevance_score(
                    query_embedding,
                    result_embedding,
                    match.metadata.get('content', '')
                )
                if relevance_score >= min_similarity:
                    match.score = relevance_score
                    filtered_results.append(match)
        except Exception as e:
            logger.error(f"Error processing match in filter_relevant_results: {e}")
            continue
    return filtered_results

def search_articles(query_text: str, top_k: int = 5) -> List[Any]:
    """
    Enhanced search for articles using the query text with translation support
    """
    try:
        # Translate query to Gujarati if it's in English
        gujarati_query = translate_to_gujarati(query_text)
        logger.info(f"Translated query: {gujarati_query}")

        # Get embedding for the Gujarati query
        query_embedding = safe_get_embedding(gujarati_query)
        if not query_embedding:
            st.error("Could not generate embedding for your query.")
            return []

        initial_top_k = min(top_k * 3, 20)
        results = index.query(
            vector=query_embedding,
            top_k=initial_top_k,
            include_metadata=True
        )

        if not results.matches:
            st.warning("No results found for your query.")
            return []

        filtered_results = filter_relevant_results(results.matches, query_embedding)

        if not filtered_results:
            st.warning("No relevant results found after filtering.")
            return []

        filtered_results.sort(key=lambda x: x.score, reverse=True)
        return filtered_results[:top_k]

    except Exception as e:
        logger.error(f"Error in search: {e}")
        st.error(f"Error in search: {e}")
        return []

def get_chat_response(messages: List[Dict[str, str]],
                     lang_pref: str = "Bilingual") -> Optional[str]:
    """
    Get response from ChatGPT with language preference
    """
    try:
        if lang_pref == "Gujarati Only":
            messages.append({"role": "system", "content": "Please respond in Gujarati only."})
        elif lang_pref == "English Only":
            messages.append({"role": "system", "content": "Please respond in English only."})

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting chat response: {e}")
        st.error(f"Error getting chat response: {e}")
        return None

def display_article(match: Any, index: int, query: str) -> bool:
    """
    Enhanced article display with relevance information
    """
    try:
        relevance_score = getattr(match, 'score', 0)
        if relevance_score >= 0.6:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**Article {index}**")
                with col2:
                    st.markdown(f"Relevance: {relevance_score:.2f}")

                content = match.metadata.get('content', 'No content available')
                st.markdown(
                    f"""<div style='background-color: #f0f2f6; padding: 10px;
                    border-radius: 5px; margin: 10px 0;'>{content}</div>""",
                    unsafe_allow_html=True
                )

                if 'url' in match.metadata:
                    st.markdown(f"[Read full article]({match.metadata['url']})")
                st.divider()
                return True
        return False
    except Exception as e:
        logger.error(f"Error displaying article: {e}")
        st.error(f"Error displaying article: {e}")
        return False

def initialize_session_state():
    """
    Initialize session state variables
    """
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in Gujarati news. You can understand and respond in both Gujarati and English. When searching, English queries will be translated to Gujarati for better results."}
        ]
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 3
    if 'lang_pref' not in st.session_state:
        st.session_state.lang_pref = "Bilingual"
    if 'relevance_threshold' not in st.session_state:
        st.session_state.relevance_threshold = 0.6

def render_sidebar():
    """
    Render sidebar elements
    """
    with st.sidebar:
        st.header("Search Options")

        st.session_state.top_k = st.slider(
            "Number of results",
            1, 10, 3,
            help="Maximum number of articles to display"
        )

        st.session_state.lang_pref = st.radio(
            "Preferred Response Language",
            ["Bilingual", "Gujarati Only", "English Only"]
        )

        st.session_state.relevance_threshold = st.slider(
            "Relevance Threshold",
            0.0, 1.0, 0.6,
            help="Minimum relevance score for articles to be displayed"
        )

        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "system", "content": "You are a helpful assistant specializing in Gujarati news. You can understand and respond in both Gujarati and English."}
            ]
            st.rerun()

        st.markdown("---")
        st.markdown("""
        ### How to use:
        1. Type your question in Gujarati or English
        2. View AI's response based on relevant news
        3. Expand 'View Related Articles' to see sources
        4. Adjust settings in this sidebar
        """)

def main():
    st.title("ગુજરાતી સમાચાર શોધ 🔍")
    st.subheader("Gujarati News Search with AI")

    initialize_session_state()
    render_sidebar()

    user_input = st.chat_input("Enter your query in Gujarati or English...")

    if user_input:
        # Translate user input if it's in English
        gujarati_query = translate_to_gujarati(user_input)

        # Show translation if query was changed
        if gujarati_query != user_input:
            st.info(f"Searching for: {gujarati_query}")

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner('Searching through news articles...'):
            results = search_articles(gujarati_query, top_k=st.session_state.top_k)

        if results:
            context = "Based on the following news articles:\n\n"
            for i, match in enumerate(results, 1):
                context += f"{i}. {match.metadata['content']}\n\n"

            context_message = {"role": "system", "content": context}
            messages = st.session_state.messages + [context_message]

            with st.spinner('Generating response...'):
                ai_response = get_chat_response(messages, st.session_state.lang_pref)

            if ai_response:
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if results:
            with st.expander("View Related Articles", expanded=False):
                displayed_count = 0
                for i, match in enumerate(results, 1):
                    if display_article(match, i, user_input):
                        displayed_count += 1

                if displayed_count == 0:
                    st.info("No highly relevant articles found for your query.")

if __name__ == "__main__":
    main()
