"""
Here we implement the german wikipedia scraper and call all related functionalities
"""

import requests
from bs4 import BeautifulSoup
import re
import csv
from collections import Counter
import spacy as spacy
import de_verb_lstm as verb
import de_adjective_lstm as adjective
import pickle as pickle
import torch as t

nlp = spacy.load('de_core_news_sm') # load the German model in - I don't want to download the large ones so we just use small here

# load the encoders
with open("./lemmatize_adj_model/encoder.pkl", "rb") as f:
        adj_Encoder:adjective.encode = pickle.load(f)
with open("./lemmatize_verb_model/encoder.pkl", "rb") as f:
        verb_Encoder:verb.encode = pickle.load(f)

custom_adj_model = adjective.Seq2SeqLSTM(len(adj_Encoder.char2idx))
custom_adj_model.load_state_dict(t.load("./lemmatize_adj_model/de_adj_lemmatizer_model.pt")) # load weights from file

custom_verb_model = verb.Seq2SeqLSTM(len(verb_Encoder.char2idx))
custom_verb_model.load_state_dict(t.load("./lemmatize_verb_model/de_verb_lemmatizer_model.pt")) # load weights from file

device = t.device("cuda" if t.cuda.is_available() else "cpu")
custom_adj_model.to(device)
custom_verb_model.to(device)

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

def gen_freq_dict(topic="Tier", model = "spaCy"):
    '''Please note that "topic" must be the name of a valid German wikipedia page
    "model" must be either "spaCy" or "custom".'''
    if type(topic) == str:
        try:
            list = list_all_words(get_sub_links("https://de.wikipedia.org/wiki/"+topic))
        except:
            raise PermissionError("Please choose a valid topic.") # it's probably not right to choose this error, but it'll get the job done
    else: # assume that the topic is iterable
         list = []
         for item in topic:
            try:
                list += list_all_words(get_sub_links("https://de.wikipedia.org/wiki/"+item))
            except:
                raise PermissionError("Please choose a valid topic.")

    lemmatized_list = []
    for word in list:
        classify = nlp(word)
        for item in classify:
            # the below would certainly be better if we used spaCy to do the lemmatization, but that would kind of defeat the point of making the AI models to begin with
            if model == "custom":
                if item.pos_ == "VERB": # if part of speech is verb
                    out = verb.lemmatize(custom_verb_model, verb_Encoder, item.text)
                elif item.pos_ == "ADJ": # is adjective
                    out = adjective.lemmatize(custom_adj_model, adj_Encoder, item.text)
                elif item.pos_ != "PUNCT": # do not allow punctuation
                    out = item.text
                
                lemmatized_list.append(out)

            elif model == "spaCy":
                if item.pos_ != "PUNCT":
                    lemmatized_list.append(item.lemma_)
                 
            else:
                 raise ValueError("Please input a valid model; either \"spaCy\" or \"custom\".")

    write_word_frequencies(lemmatized_list,"Tier_word_frequency")

    # now we remove the words that are the top 1000 most common in the language at large so that the remaining vocabulary is somewaht specific to the discipline at hand.


if __name__ == "__main__":

    gen_freq_dict()