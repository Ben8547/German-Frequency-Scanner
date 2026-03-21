"""
Here we implement the german wikipedia scraper and call all related functionalities
"""

import requests
from bs4 import BeautifulSoup
import re
import csv
from collections import Counter
from Verb_Neural_Net import load_verb_lemmatizer, lemmatize

def get_sub_links(url): # this function returns all of the sublinks to  a given page - this is the main method that we will use to determine which related topics to investigate
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


def list_all_words(list_links): # 
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


verb_model, verb_tokenizer = load_verb_lemmatizer()
print(lemmatize("ausgekommen",verb_model,verb_tokenizer))