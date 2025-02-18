import streamlit as st
from huggingface_hub import InferenceClient
from deep_translator import GoogleTranslator
import json
from datetime import datetime, timedelta
import os
import re
from itertools import combinations

# Initialize translators
def initialize_translators():
    en_to_gu = GoogleTranslator(source='english', target='gujarati')
    gu_to_en = GoogleTranslator(source='gujarati', target='english')
    return en_to_gu, gu_to_en

def detect_language(text):
    # Simple detection based on character set
    gujarati_pattern = re.compile(r'[\u0A80-\u0AFF]')
    if gujarati_pattern.search(text):
        return 'gujarati'
    return 'english'

def translate_text(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text

    translator = GoogleTranslator(source=source_lang, target=target_lang)
    try:
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

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

def extract_tags_and_date(query, client, source_lang='english'):
    query_en = query if source_lang == 'english' else translate_text(query, 'gujarati', 'english')

    prompt = f"""
    Extract relevant search tags and date range from the following query: "{query_en}"
    Return the response in JSON format with keys 'tags' and 'date_range'.
    If no date is mentioned, set date_range to null.
    Examples:
    - For "cricket news in last 5 days" return {{"tags": ["cricket"], "date_range": 5}}
    - For "road accidents in gujarat" return {{"tags": ["road", "accident", "gujarat"], "date_range": null}}
    - For "latest political news" return {{"tags": ["political"], "date_range": null}}
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
            return result['tags'], result.get('date_range', None)
        except json.JSONDecodeError:
            return extract_tags_with_regex(query_en)
    except Exception as e:
        st.error(f"Error calling Hugging Face API: {str(e)}")
        return extract_tags_with_regex(query_en)

def extract_tags_with_regex(query):
    date_range = None
    if 'day' in query.lower() or 'days' in query.lower():
        match = re.search(r'(\d+)\s*days?', query.lower())
        if match:
            date_range = int(match.group(1))
    elif 'week' in query.lower():
        date_range = 7
    elif 'month' in query.lower():
        date_range = 30

    words = query.lower().split()
    stop_words = {'news', 'last', 'days', 'week', 'month', 'the', 'in', 'on', 'at', 'from', 'to', 'સમાચાર', 'છેલ્લા'}
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
    start_date = current_date - timedelta(days=date_range) if date_range else None

    try:
        for filename in os.listdir('news_articles'):
            if filename.endswith('.txt'):
                with open(os.path.join('news_articles', filename), 'r', encoding='utf-8') as f:
                    content = f.read()

                    date_match = re.search(r'Date: (\d{2}-\d{2}-\d{4})', content)
                    if date_match:
                        article_date = datetime.strptime(date_match.group(1), '%d-%m-%Y')

                        if start_date and not (start_date <= article_date <= current_date):
                            continue

                        tag_combinations = get_all_tag_combinations(tags)
                        for tag_combo in tag_combinations:
                            if tag_combo.lower() in content.lower():
                                title_match = re.search(r'Title: (.*?)\n', content)
                                link_match = re.search(r'Link: (.*?)\n', content)

                                title = title_match.group(1) if title_match else "No title"
                                link = link_match.group(1) if link_match else ""

                                # Highlight tags in content
                                highlighted_content = content
                                for tag in tags:
                                    pattern = re.compile(re.escape(tag), re.IGNORECASE)
                                    highlighted_content = pattern.sub(f"**{tag}**", highlighted_content)

                                results.append({
                                    'title': title,
                                    'content': highlighted_content,
                                    'date': article_date.strftime('%d-%m-%Y'),
                                    'link': link
                                })
                                break

    except Exception as e:
        st.error(f"Error reading news files: {str(e)}")
        return []

    return results

def main():
    st.title("Bilingual News Search Bot")

    @st.cache_resource
    def load_resources():
        client = initialize_client()
        en_to_gu, gu_to_en = initialize_translators()
        return client, en_to_gu, gu_to_en

    client, en_to_gu, gu_to_en = load_resources()

    output_lang = st.radio("Select output language:", ['English', 'Gujarati'])

    query = st.text_input("Enter your query in English or Gujarati (દાખલા તરીકે: 'ક્રિકેટ સમાચાર' અથવા 'cricket news')")

    if query:
        with st.spinner("Processing query..."):
            input_lang = detect_language(query)

            tags, date_range = extract_tags_and_date(
                query,
                client,
                source_lang='gujarati' if input_lang == 'gujarati' else 'english'
            )

            # Always translate tags to Gujarati for search
            gujarati_tags = [translate_text(tag, 'english', 'gujarati') for tag in tags]

            # Show extracted information
            if output_lang == 'Gujarati':
                st.write("શોધેલા ટૅગ્સ:", gujarati_tags)
                st.write("તારીખ સીમા:", f"છેલ્લા {date_range} દિવસ" if date_range else "તમામ લેખો")
            else:
                st.write("Extracted Tags:", tags)
                st.write("Date Range:", f"Last {date_range} days" if date_range else "All articles")

            # Search using Gujarati tags
            results = search_news_articles(gujarati_tags, date_range)

            if results:
                if output_lang == 'Gujarati':
                    st.write(f"કુલ {len(results)} લેખો મળ્યા:")
                else:
                    st.write(f"Found {len(results)} relevant articles:")

                for idx, article in enumerate(results):
                    title = article['title']
                    content = article['content']

                    # Translate if needed
                    if output_lang == 'English' and input_lang == 'gujarati':
                        title = translate_text(title, 'gujarati', 'english')
                        content = translate_text(content, 'gujarati', 'english')
                    elif output_lang == 'Gujarati' and input_lang == 'english':
                        title = translate_text(title, 'english', 'gujarati')
                        content = translate_text(content, 'english', 'gujarati')

                    with st.expander(f"{title} ({article['date']})"):
                        st.markdown(content)  # Using markdown to render bold text
                        if article['link']:
                            link_text = "લેખની લિંક" if output_lang == 'Gujarati' else "Link to Article"
                            st.markdown(f"[{link_text}]({article['link']})")
            else:
                if output_lang == 'Gujarati':
                    st.write("કોઈ લેખ મળ્યા નથી.")
                else:
                    st.write("No articles found matching your query.")

if __name__ == "__main__":
    main()
    
