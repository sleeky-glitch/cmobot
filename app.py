import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
import numpy as np

# Initialize OpenAI and Pinecone clients
@st.cache_resource
def init_clients():
    try:
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        return openai_client, pc
    except Exception as e:
        st.error(f"Error initializing clients: {e}")
        st.stop()

# Initialize the clients
openai_client, pc = init_clients()

# Get Pinecone index
try:
    index_name = st.secrets.get("PINECONE_INDEX_NAME", "gujarati-news")
    index = pc.Index(index_name)
except Exception as e:
    st.error(f"Error connecting to Pinecone index: {e}")
    st.stop()

def search_articles(query_text, top_k=5):
    """
    Search for articles using the query text
    """
    try:
        # Get embedding for the query
        response = openai_client.embeddings.create(
            input=query_text,
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding

        # Search in Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        return results.matches

    except Exception as e:
        st.error(f"Error in search: {e}")
        return None

def get_chat_response(messages, lang_pref="Bilingual"):
    """
    Get response from ChatGPT with language preference
    """
    try:
        # Add language preference instruction
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
        st.error(f"Error getting chat response: {e}")
        return None

def display_article(match, index):
    """
    Display a single article with formatting
    """
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**Article {index}**")
    with col2:
        st.markdown(f"Score: {match.score:.2f}")
    st.write(match.metadata['content'])
    if 'url' in match.metadata:
        st.markdown(f"[Read full article]({match.metadata['url']})")
    st.divider()

def initialize_session_state():
    """
    Initialize session state variables
    """
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in Gujarati news. You can understand and respond in both Gujarati and English. Provide concise and relevant responses based on the news articles provided."}
        ]
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 3
    if 'lang_pref' not in st.session_state:
        st.session_state.lang_pref = "Bilingual"

def render_sidebar():
    """
    Render sidebar elements
    """
    with st.sidebar:
        st.header("Search Options")

        # Number of results slider
        st.session_state.top_k = st.slider("Number of results", 1, 10, 3)

        # Language preference
        st.session_state.lang_pref = st.radio(
            "Preferred Response Language",
            ["Bilingual", "Gujarati Only", "English Only"]
        )

        # Clear chat button
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

        ### Examples:
        - "ગુજરાતમાં તાજેતરની મુખ્ય ઘટનાઓ શું છે?"
        - "What are the recent developments in Gujarat?"
        - "ગુજરાતમાં શિક્ષણ ક્ષેત્રે થયેલા સુધારા"
        """)

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app searches through a database of Gujarati news articles
        using AI to provide relevant information and insights.

        Built with Streamlit, OpenAI, and Pinecone.
        """)

def main():
    st.title("ગુજરાતી સમાચાર શોધ 🔍")
    st.subheader("Gujarati News Search with AI")

    # Initialize session state
    initialize_session_state()

    # Render sidebar
    render_sidebar()

    # Chat input
    user_input = st.chat_input("Enter your query in Gujarati or English...")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Search for relevant articles
        with st.spinner('Searching through news articles...'):
            results = search_articles(user_input, top_k=st.session_state.top_k)

        if results:
            # Prepare context from search results
            context = "Based on the following news articles:\n\n"
            for i, match in enumerate(results, 1):
                context += f"{i}. {match.metadata['content']}\n\n"

            # Add context to the conversation
            context_message = {"role": "system", "content": context}
            messages = st.session_state.messages + [context_message]

            # Get AI response
            with st.spinner('Generating response...'):
                ai_response = get_chat_response(messages, st.session_state.lang_pref)

            if ai_response:
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

        # Display chat history
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Display related articles in expander
        if results:
            with st.expander("View Related Articles", expanded=False):
                for i, match in enumerate(results, 1):
                    display_article(match, i)

if __name__ == "__main__":
    main()
