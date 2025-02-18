import streamlit as st
from huggingface_hub import InferenceClient
import json
from datetime import datetime, timedelta
import os
import re
from itertools import combinations

# Initialize Hugging Face client
def initialize_client():
    try:
        client = InferenceClient(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            token=st.secrets["huggingface_token"]
        )
        return client
    except Exception as e:
        st.error(f"Error initializing Hugging Face client: {str(e)}")
        st.error("Please ensure you have set up your secrets.toml file correctly.")
        return None

def extract_tags_and_date(query, client):
    if not client:
        return [], 7

    prompt = f"""
    Extract relevant search tags and date range from the following query: "{query}"
    Return the response in JSON format with keys 'tags' and 'date_range'.
    Example: For "cricket news in last 5 days" return {{"tags": ["cricket"], "date_range": 5}}
    For "road accidents in gujarat last week" return {{"tags": ["road", "accident", "gujarat"], "date_range": 7}}
    """

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=200,
            temperature=0.3,
            return_full_text=False
        )

        try:
            result = json.loads(response)
            return result['tags'], result['date_range']
        except json.JSONDecodeError:
            # Fallback to regex extraction
            return extract_tags_with_regex(query)
    except Exception as e:
        st.error(f"Error calling Hugging Face API: {str(e)}")
        return extract_tags_with_regex(query)

def extract_tags_with_regex(query):
    # Default date range is 7 days
    date_range = 7

    # Extract date range from query
    if 'day' in query.lower() or 'days' in query.lower():
        match = re.search(r'(\d+)\s*days?', query.lower())
        if match:
            date_range = int(match.group(1))
    elif 'week' in query.lower():
        date_range = 7
    elif 'month' in query.lower():
        date_range = 30

    # Extract potential tags
    words = query.lower().split()
    stop_words = {'news', 'last', 'days', 'week', 'month', 'the', 'in', 'on', 'at', 'from', 'to'}
    tags = [word for word in words if len(word) > 3 and word not in stop_words]

    return tags, date_range

def get_all_tag_combinations(tags):
    all_combinations = []
    for r in range(1, len(tags) + 1):
        all_combinations.extend(combinations(tags, r))
    return [' '.join(combo) for combo in all_combinations]

def search_news_articles(tags, date_range):
    results = []
    current_date = datetime.now()
    start_date = current_date - timedelta(days=date_range)

    try:
        # Read all news files in the directory
        for filename in os.listdir('news_articles'):
            if filename.endswith('.txt'):
                with open(os.path.join('news_articles', filename), 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Extract article date
                    date_match = re.search(r'Date: (\d{2}-\d{2}-\d{4})', content)
                    if date_match:
                        article_date = datetime.strptime(date_match.group(1), '%d-%m-%Y')

                        # Check if article is within date range
                        if start_date <= article_date <= current_date:
                            # Search for all tag combinations
                            tag_combinations = get_all_tag_combinations(tags)
                            for tag_combo in tag_combinations:
                                if tag_combo.lower() in content.lower():
                                    # Extract title and link
                                    title_match = re.search(r'Title: (.*?)\n', content)
                                    link_match = re.search(r'Link: (.*?)\n', content)

                                    title = title_match.group(1) if title_match else "No title"
                                    link = link_match.group(1) if link_match else ""

                                    results.append({
                                        'title': title,
                                        'content': content,
                                        'date': article_date.strftime('%d-%m-%Y'),
                                        'link': link
                                    })
                                    break  # Avoid duplicate articles

    except Exception as e:
        st.error(f"Error reading news files: {str(e)}")
        return []

    return results

def main():
    st.title("Gujarati News Search Bot")

    # Initialize client (with caching)
    @st.cache_resource
    def load_client():
        return initialize_client()

    client = load_client()

    # User input
    query = st.text_input("Enter your news query (e.g., 'cricket news in last 5 days')")

    if query:
        with st.spinner("Processing query..."):
            # Extract tags and date range
            tags, date_range = extract_tags_and_date(query, client)

            # Show extracted information
            st.write("Extracted Tags:", tags)
            st.write("Date Range:", f"Last {date_range} days")

            # Search for articles
            results = search_news_articles(tags, date_range)

            # Display results
            if results:
                st.write(f"Found {len(results)} relevant articles:")
                for idx, article in enumerate(results):
                    with st.expander(f"{article['title']} ({article['date']})"):
                        st.write(article['content'])
                        if article['link']:
                            st.markdown(f"[Read full article]({article['link']})")
            else:
                st.write("No articles found matching your query.")

if __name__ == "__main__":
    main()
