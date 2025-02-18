import os
import streamlit as st
from deep_translator import GoogleTranslator
import openai
import glob

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to load articles from text files
def load_articles():
    articles = []
    try:
        for file in glob.glob("*.txt"):  # Adjust path if needed
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                # Extract title, date, link, and content
                try:
                    title = content.split("Title: ")[1].split("\n")[0]
                    date = content.split("Date: ")[1].split("\n")[0]
                    link = content.split("Link: ")[1].split("\n")[0]
                    article_content = content.split("Content: ")[1]
                    articles.append({
                        "title": title,
                        "date": date,
                        "link": link,
                        "content": article_content
                    })
                except IndexError:
                    st.warning(f"Could not parse file: {file}")
    except Exception as e:
        st.error(f"Error loading articles: {e}")
    return articles

# Function to search articles using OpenAI embeddings
def search_articles(query, articles):
    try:
        # Get embedding for the query
        query_response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = query_response['data'][0]['embedding']

        # Get embeddings for all articles
        results = []
        for article in articles:
            article_text = f"{article['title']} {article['content']}"
            article_response = openai.Embedding.create(
                input=article_text,
                model="text-embedding-ada-002"
            )
            article_embedding = article_response['data'][0]['embedding']

            # Calculate similarity
            similarity = cosine_similarity(query_embedding, article_embedding)
            if similarity > 0.7:  # Adjust threshold as needed
                results.append((article, similarity))

        # Sort results by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results]
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    import numpy as np
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Function to translate Gujarati to English
def translate_to_english(text):
    try:
        translated_text = GoogleTranslator(source="gujarati", target="english").translate(text)
        return translated_text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return text

# Streamlit app
def main():
    st.title("Gujarati News Search and Translation Bot")
    st.write("Search for Gujarati news articles and optionally translate them to English.")

    # Initialize session state for articles
    if 'articles' not in st.session_state:
        st.session_state.articles = load_articles()

    # Add a refresh button
    if st.button("Refresh Articles"):
        st.session_state.articles = load_articles()

    # User input for search query
    query = st.text_input("Enter your search query (in English or Gujarati):")

    if query:
        # Search articles
        results = search_articles(query, st.session_state.articles)

        if results:
            st.write(f"Found {len(results)} relevant articles:")
            for i, article in enumerate(results):
                with st.expander(f"{i+1}. {article['title']}"):
                    st.write(f"**Date:** {article['date']}")
                    st.write(f"**Link:** [Read more]({article['link']})")

                    # Create two columns for original and translated content
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Original Content:**")
                        st.write(article['content'])

                    with col2:
                        st.write("**English Translation:**")
                        if st.button(f"Translate Article {i+1}", key=f"translate_{i}"):
                            translated_content = translate_to_english(article["content"])
                            st.write(translated_content)
        else:
            st.write("No articles found for your query.")

    # Add footer with information
    st.markdown("---")
    st.markdown("""
    ### About this app
    - Search through Gujarati news articles
    - Translate content to English
    - Powered by OpenAI and Google Translate
    """)

if __name__ == "__main__":
    main()
