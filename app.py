import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from itertools import combinations
import os
import json
from datetime import datetime, timedelta
import re

# Initialize Mixtral
def initialize_model():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def extract_tags_and_date(query, model, tokenizer):
    prompt = f"""
    Extract relevant search tags and date range from the following query: "{query}"
    Return the response in JSON format with keys 'tags' and 'date_range'.
    Example: For "cricket news in last 5 days" return {{"tags": ["cricket"], "date_range": 5}}
    For "road accidents in gujarat last week" return {{"tags": ["road", "accident", "gujarat"], "date_range": 7}}
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        result = json.loads(response)
        return result['tags'], result['date_range']
    except:
        return [], 7  # default to week if parsing fails

def get_all_tag_combinations(tags):
    all_combinations = []
    for r in range(1, len(tags) + 1):
        all_combinations.extend(combinations(tags, r))
    return [' '.join(combo) for combo in all_combinations]

def search_news_articles(tags, date_range):
    results = []
    current_date = datetime.now()
    start_date = current_date - timedelta(days=date_range)

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
                                # Extract title
                                title_match = re.search(r'Title: (.*?)\n', content)
                                title = title_match.group(1) if title_match else "No title"

                                results.append({
                                    'title': title,
                                    'content': content,
                                    'date': article_date.strftime('%d-%m-%Y')
                                })
                                break  # Avoid duplicate articles

    return results

def main():
    st.title("Gujarati News Search Bot")

    # Initialize model (you might want to cache this)
    @st.cache_resource
    def load_model():
        return initialize_model()

    model, tokenizer = load_model()

    # User input
    query = st.text_input("Enter your news query (e.g., 'cricket news in last 5 days')")

    if query:
        with st.spinner("Processing query..."):
            # Extract tags and date range
            tags, date_range = extract_tags_and_date(query, model, tokenizer)

            # Search for articles
            results = search_news_articles(tags, date_range)

            # Display results
            if results:
                st.write(f"Found {len(results)} relevant articles:")
                for idx, article in enumerate(results):
                    with st.expander(f"{article['title']} ({article['date']})"):
                        st.write(article['content'])
            else:
                st.write("No articles found matching your query.")

if __name__ == "__main__":
    main()

# Created/Modified files during execution:
# None (reads from existing news_articles/*.txt files)
