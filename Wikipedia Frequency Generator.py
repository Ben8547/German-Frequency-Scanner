import requests
from bs4 import BeautifulSoup
import re
import csv
from collections import Counter

def get_sub_links(url):
    # request 
    response = requests.get(
        url=url,
    )
    # soup object
    soup = BeautifulSoup(response.content, 'html.parser')

    main_article = soup.find(class_ = "mw-content-ltr mw-parser-output")

    allLinks = main_article.find_all("a", href=True)

    linkToScrape = []
    # find sub articles
    for link in allLinks:
        # Use this link to scrape
        linkToScrape.append("https://de.wikipedia.org/" + link['href'])
        #print("https://de.wikipedia.org/" + link['href'])
    linkToScrape.append(url)
    return linkToScrape

def list_all_words(list_links):
    word_list = []
    for link in list_links:
        sub_word_list=[]
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        main_article = soup.find(class_="mw-content-ltr mw-parser-output")
        # attempt to troubleshoot variation in links
        if main_article is None:
            main_article = soup.find(class_="mw-page-container-inner")
            if main_article is None:
                print("Could not find main article content: %s"%link)
                sub_word_list = []
                continue

        # Extract text
        text = main_article.get_text(separator=' ', strip=True)

        # Lowercase, remove other symbols
        sub_word_list = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        word_list += sub_word_list
    return word_list

def write_word_frequencies(word_list, output_filename="word_frequencies.csv"):
    # Count word frequencies
    word_counts = Counter(word_list)

    # Sort by frequency (descending)
    sorted_word_counts = word_counts.most_common()

    # Write to CSV
    with open(output_filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['word', 'frequency'])  # Header
        for word, freq in sorted_word_counts:
            writer.writerow([word, freq])

    print(f"Wrote {len(sorted_word_counts)} word frequencies to {output_filename}")

#print(get_sub_links("https://de.wikipedia.org/wiki/Tier"))
#print(list_all_words(get_sub_links("https://de.wikipedia.org/wiki/Tier")))
list = list_all_words(get_sub_links("https://de.wikipedia.org/wiki/Tier")) + list_all_words(get_sub_links("https://de.wikipedia.org/wiki/Pflanze"))
write_word_frequencies(list,"Tier_word_frequency")