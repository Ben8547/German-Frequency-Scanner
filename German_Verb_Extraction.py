'''
This script aims to create a database of a large number of german verbs, and their possible conjucations. This in pusuit of training a model to 
map conjugated verbs to their dictionary form.

Steps:

1. Make a master list of german verbs in their dictionary form
2. Look up the verbs in a ductionary to extract their conjugated forms
3. create a database that tags each conjugated form with its dictionary form - this will be the training data for the AI model.

'''

import requests
from bs4 import BeautifulSoup # for webscraping
from time import sleep
from numpy.random import choice, random
import re
from numpy import savetxt, array
import csv
from fake_useragent import UserAgent # hopefully solve the issues with rate blocking on verbformen


######
# Step 1 - download a master list of german verbs
######

HEADERS = {
    "User-Agent": (UserAgent().chrome)} # we need the header because wikitionary blocks some bot-traffic

def get_verbs(lang = 'de'):

    if lang == 'de':
        BASE_URL = "https://en.wiktionary.org"
        START_URL = "https://en.wiktionary.org/wiki/Category:German_verbs"

        url = START_URL
        verbs = set()

        session = requests.Session()
        session.headers.update(HEADERS)

        while url:
            print(f"Scraping: {url}") # progress update
            res = session.get(url)
            res.raise_for_status()

            soup = BeautifulSoup(res.text, "html.parser")

            mw_pages = soup.find("div", id="mw-pages") # Find the category listing
            if not mw_pages:
                break

            for li in mw_pages.find_all("li"): # Extract all verbs (links inside list items) on given page
                a = li.find("a")
                if a and a.text:
                    # now remove multi-word verbs
                    if a and a.text and " " not in a.text:
                        verbs.add(a.text)

            # Find next page link
            next_link = None
            for a in mw_pages.find_all("a"):
                if a.text.lower() == "next page":
                    next_link = BASE_URL + a["href"]
                    break

            url = next_link

            sleep(0.001) # server curtosey - wikipedia is probably built to handle the traffic, but its a good practice nonetheless I think.
    else:
        raise ValueError("Please input a supported language")

    return verbs

def sample_percent(verb_set:set, p:float): # the list of verbs is very long, and there would be no point to training the model on all of them as then we could just built a look-up table, and storing ~9,000 strings is expensize anyway. Thus we can store about 20% of them as the training data.
    l = len(verb_set)
    n = round(p*l)
    return choice(list(verb_set),n, replace = False)


def is_valid_verb_form(word):
    # Keep only single "word-like" tokens (no sentences, no punctuation blobs)
    return re.fullmatch(r"[a-zäöüß]+", word) is not None


def get_conjugations_de(verb, session):
    BASE_URL = "https://www.verbformen.com/conjugation/?w="
    url = BASE_URL + verb

    response = session.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed for {verb}, status code {response.status_code}")
        if response.status_code == 429:
            print("Rate limited, Waiting.")
            sleep(60)
        return set()
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    forms = set()

    # 🔑 ONLY extract from conjugation tables
    tables = soup.select("div.vTbl")
    #print(tables[0]) # debug

    for table in tables:
        for form in table.select("td"): # the conjugated forms are inside of <u></u> tags
            text = form.text
            if "(" not in text:
                text = re.sub(r"[^a-zäöüß ]", "", text.lower()) # clean the string
                text = re.sub(r"^ ", "", text) # remove leading spaces
                if text not in {'', "ich", "du", "ihr", "wir", "er", "sie", "es", "Sie", " ich", " du", " ihr", " wir", " er", " sie", " es", " Sie"} : # prevent the pronouns from being included
                    if (verb == 'haben') or text not in {"habe", "hast", "hat", "haben", "habt", "hatte", "hattest", 'hatte', "hatten", "habet", "habest", "hattet", "hätte", "hättest", "hätten", "hättet"}:
                        if (verb == 'sein') or text not in {"sein", "bist", "bin", "seid", "ist"}:
                            if (verb == "werden") or text not in {'werde', "wirst", "wird", "werden", "werdet", "werdest", "würdest", "würden", "würdet", "würdet", "würde", "wollte", ""}:
                                #print(text) # debug
                                forms.add(text)

    return forms

def save_as_csv_de(set_of_verbs:set, file_name:str):

    savetxt(file_name,array(["# form","root"]), fmt="%s", delimiter=',') # create the headers

    HEADERS = {
            "User-Agent": (UserAgent().chrome)}
    session = requests.Session()
    session.headers.update(HEADERS) # we want to randomize the header - hopefully this staves off disconnections.

    with open(file_name,'a',newline='\n', encoding='utf-8') as file: # mode a is append mode - if we get blocked from the site, this can ensure the progress thus far is saved
        writer = csv.writer(file)
        for v in set_of_verbs:

            print(f"current verb: {v}")
            sleep(3 + random() * 7) # sleep from 3 - 10 seconds, hopefully prevents bans, though will take a while to download
            forms = get_conjugations_de(v, session)
            for form in forms:
                writer.writerow([form, v])
                file.flush() # writes to file
                #savetxt(file, [form, v], fmt='%s', delimiter=',') 




if __name__ == "__main__":

    verbs = get_verbs()
    verbs = sample_percent(verbs, 0.1)
    save_as_csv_de(verbs,"German_Verb_Training_Data.csv")

    
    # tests
    #session = requests.Session()
    #session.headers.update(HEADERS)
    #verbs = get_verbs()
    #print(f"\nTotal verbs found: {len(verbs)}")
    #print(get_conjugations_de("hypen",session))
    #print(verbs)
    pass