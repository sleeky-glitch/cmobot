import streamlit as st
from huggingface_hub import InferenceClient
from deep_translator import GoogleTranslator
import os
import re
from datetime import datetime, timedelta
import json

# Initialize Hugging Face client
@st.cache_resource
def get_hf_client():
    HUGGING_FACE_TOKEN = st.secrets["HUGGING_FACE_TOKEN"]  # Store your token in Streamlit secrets
    return InferenceClient(token=HUGGING_FACE_TOKEN)

# Initialize the client
client = get_hf_client()

# Helper function for LLM inference
def get_mixtral_response(prompt):
    try:
        response = client.text_generation(
            prompt,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.95,
            return_full_text=False
        )
        return response
    except Exception as e:
        st.error(f"Error calling Mixtral: {str(e)}")
        return None

# Helper function to translate text
def translate_text(text, source_lang, target_lang):
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

# Helper function to extract tags and date from user input
def extract_tags_and_date(user_input):
    prompt = f"""Extract relevant search tags and date from the following text.
    If a specific date is not mentioned, consider the last 5 days.
    Return the response in JSON format with 'tags' and 'date' fields.

    Text: {user_input}

    Example response format:
    {{"tags": ["cricket", "match"], "date": "2025-02-18"}}
    """

    response = get_mixtral_response(prompt)
    st.write("Raw response from Mixtral:", response)
    try:
        # Extract JSON from the response
        json_str = re.search(r'\{.*\}', response)
        if json_str:
            result = json.loads(json_str.group())
            return result.get('tags', []), result.get('date')
    except Exception as e:
        st.error(f"Error parsing LLM response: {str(e)}")

    return [], None

# Helper function to search articles
def search_articles(tags, date=None):
    results = []
    files = [f for f in os.listdir() if f.startswith("dd_news_page") and f.endswith(".txt")]

    # Create all possible combinations of tags
    from itertools import combinations
    all_combinations = []
    for r in range(len(tags), 0, -1):
        all_combinations.extend(list(combinations(tags, r)))

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            title = re.search(r"Title:\s*(.*)", content).group(1)
            article_date_str = re.search(r"Date:\s*(.*)", content).group(1)
            article_content = re.search(r"Content:\s*(.*)", content, re.DOTALL).group(1)

            # Check date if specified
            if date:
                try:
                    article_date = datetime.strptime(article_date_str.split("|")[0].strip(), "%d-%m-%Y")
                    if article_date < date - timedelta(days=5) or article_date > date:
                        continue
                except ValueError:
                    continue

            # Check tag combinations
            for combo in all_combinations:
                if all(tag.lower() in article_content.lower() for tag in combo):
                    score = len(combo)  # Score based on number of matching tags
                    results.append({
                        'title': title,
                        'content': article_content,
                        'date': article_date_str,
                        'score': score,
                        'matching_tags': combo
                    })
                    break  # Move to next article once we find a matching combination

    # Sort results by score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

# Streamlit UI
st.title("Gujarati News Search Bot")

# User input
user_input = st.text_input("Ask a question (e.g., 'Give me cricket news in the last five days'):")
translate_to_english = st.checkbox("Translate results to English", value=False)

if user_input:
    with st.spinner("Processing your request..."):
        # Translate user input to Gujarati if it's in English
        user_input_gujarati = translate_text(user_input, "en", "gu") if re.search(r"[a-zA-Z]", user_input) else user_input

        # Extract tags and date
        tags, date_str = extract_tags_and_date(user_input_gujarati)

        if tags:
            st.write("📌 Searching for articles with these tags:", ", ".join(tags))

            # Convert date string to datetime object
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime.now()
            except ValueError:
                date = datetime.now()

            # Perform search
            results = search_articles(tags, date)

            if results:
                st.write(f"Found {len(results)} relevant articles:")
                for idx, result in enumerate(results, 1):
                    with st.expander(f"{idx}. {result['title']} ({result['date']})"):
                        if translate_to_english:
                            translated_content = translate_text(result['content'], "gu", "en")
                            st.write(translated_content)
                        else:
                            st.write(result['content'])
                        st.write(f"Matching tags: {', '.join(result['matching_tags'])}")
            else:
                st.warning("No articles found matching your search criteria.")
        else:
            st.error("Could not extract search tags from your input. Please try rephrasing your question.")

# Add some helpful information at the bottom
st.markdown("---")
st.markdown("""
### How to use this bot:
1. Type your question in English or Gujarati
2. Optionally check the 'Translate results to English' box
3. The bot will search for relevant news articles
4. Click on any article title to read the full content
""")
