import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

def calculate_relevance_score(query_embedding, result_embedding, content):
    """
    Calculate a combined relevance score based on embedding similarity and content analysis
    """
    # Calculate cosine similarity between query and result embeddings
    similarity = cosine_similarity(
        np.array(query_embedding).reshape(1, -1),
        np.array(result_embedding).reshape(1, -1)
    )[0][0]

    return similarity

def filter_relevant_results(results, query_embedding, min_similarity=0.7):
    """
    Filter results based on relevance criteria
    """
    filtered_results = []

    for match in results:
        relevance_score = calculate_relevance_score(
            query_embedding,
            match.values,
            match.metadata['content']
        )

        if relevance_score >= min_similarity:
            match.score = relevance_score
            filtered_results.append(match)

    return filtered_results

def search_articles(query_text, top_k=5):
    """
    Enhanced search for articles using the query text with better relevance filtering
    """
    try:
        # Get embedding for the query
        response = openai_client.embeddings.create(
            input=query_text,
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding

        # Search in Pinecone with higher initial top_k to allow for filtering
        initial_top_k = min(top_k * 3, 20)  # Get more results initially
        results = index.query(
            vector=query_embedding,
            top_k=initial_top_k,
            include_metadata=True
        )

        # Filter results for relevance
        filtered_results = filter_relevant_results(results.matches, query_embedding)

        # Sort by relevance score and take top_k
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        return filtered_results[:top_k]

    except Exception as e:
        st.error(f"Error in search: {e}")
        return None

def analyze_article_relevance(query, article_content):
    """
    Analyze how relevant an article is to the query
    """
    try:
        messages = [
            {"role": "system", "content": "Analyze if the following article is relevant to the query. Respond with a score between 0 and 1."},
            {"role": "user", "content": f"Query: {query}\nArticle: {article_content}\n\nIs this article relevant? Provide a score between 0 and 1."}
        ]

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=50
        )

        try:
            score = float(response.choices[0].message.content.strip())
            return min(max(score, 0), 1)
        except:
            return 0.5
    except Exception as e:
        st.error(f"Error analyzing relevance: {e}")
        return 0.5

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

def display_article(match, index, query):
    """
    Enhanced article display with relevance information
    """
    relevance_score = analyze_article_relevance(query, match.metadata['content'])

    if relevance_score >= 0.6:  # Only display articles with good relevance
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"**Article {index}**")
        with col2:
            st.markdown(f"Similarity: {match.score:.2f}")
        with col3:
            st.markdown(f"Relevance: {relevance_score:.2f}")

        # Display article content with highlighting
        content = match.metadata['content']
        st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>{content}</div>",
                   unsafe_allow_html=True)

        if 'url' in match.metadata:
            st.markdown(f"[Read full article]({match.metadata['url']})")
        st.divider()
        return True
    return False

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

        # Relevance threshold slider
        st.session_state.relevance_threshold = st.slider(
            "Relevance Threshold",
            0.0, 1.0, 0.6,
            help="Minimum relevance score for articles to be displayed"
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
                    st.markdown(message["content"])

        # Display related articles in expander
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
