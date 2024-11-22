import wikipediaapi
from pybraille import convertText
import re
import os
import argparse
from tqdm import tqdm

# Initialize Wikipedia API with User-Agent
user_agent = "MyWikipediaApp/1.0 (https://example.com; contact@example.com)"
wiki_wiki = wikipediaapi.Wikipedia('en', headers={"User-Agent": user_agent})

# Function to fetch Wikipedia article content
def fetch_wikipedia_article(title):
    page = wiki_wiki.page(title)
    if not page.exists():
        return None
    return page.text

# Function to filter text for Braille conversion
def filter_supported_characters(text):
    filtered_text = ""
    for char in text:
        try:
            # Test if character can be converted
            braille_char = convertText(char)
            if braille_char is not None:
                filtered_text += char
        except Exception:
            pass
    return filtered_text

# Function to clean text (e.g., remove unsupported symbols)
def clean_text_for_braille(text):
    return re.sub(r'[^a-zA-Z0-9 .,!?\'\"\-\n]', '', text)

# Function to convert text to Braille
def convert_to_braille(text):
    braille_text = ""
    for char in text:
        braille_char = convertText(char)
        if braille_char is not None:
            braille_text += braille_char
        else:
            braille_text += "⍰"  # Use placeholder for unsupported characters
    return braille_text

# Function to save content to a file
def save_to_file(filename, content):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)

# Function to fetch the list of featured articles
def fetch_featured_articles(limit=500):
    featured_articles = []
    category = wiki_wiki.page("Category:Featured_articles")
    for article in category.categorymembers.values():
        if len(featured_articles) >= limit:
            break
        featured_articles.append(article.title)
    return featured_articles

# Function to sanitize filenames by replacing unsafe characters with underscores
def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

# Main workflow
def process_articles_to_braille(titles):
    # Create data directories if they don't exist
    original_dir = os.path.join('data', 'original')
    braille_dir = os.path.join('data', 'braille')
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(braille_dir, exist_ok=True)

    for index, title in enumerate(tqdm(titles, desc="Processing articles"), start=1):
        print(f"Processing article: {title}")
        sanitized_title = sanitize_filename(title)
        padded_index = f"{index:03d}"  # Pad the index with leading zeros (e.g., 001, 002, 003)
        article_content = fetch_wikipedia_article(title)
        if article_content:
            # Save original article content
            original_file = os.path.join(original_dir, f"{padded_index}_{sanitized_title}_original.txt")
            save_to_file(original_file, article_content)
            
            # Filter and convert to Braille
            filtered_content = filter_supported_characters(article_content)
            braille_content = convert_to_braille(filtered_content)
            braille_file = os.path.join(braille_dir, f"{padded_index}_{sanitized_title}_braille.txt")
            save_to_file(braille_file, braille_content)

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Process Wikipedia articles to Braille.")
    parser.add_argument('--limit', type=int, default=500, help='Number of featured articles to process')
    args = parser.parse_args()

    featured_titles = fetch_featured_articles(limit=args.limit)
    process_articles_to_braille(featured_titles)

if __name__ == "__main__":
    main()
