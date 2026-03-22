import pandas as pd
import csv
import pickle
from tqdm import tqdm # for progress bar

import requests
from bs4 import BeautifulSoup # for webscraping
from time import sleep
from numpy.random import choice, random
import re
from numpy import savetxt, array
from fake_useragent import UserAgent


######
# Step 1 - download a master list of german verbs
######

HEADERS = {"User-Agent": (UserAgent().chrome)} # we need the header because wikitionary blocks some bot-traffic

def get_adjectives(ratio_to_store:float = 1.):

    BASE_URL = "https://en.wiktionary.org"
    START_URL = "https://en.wiktionary.org/w/index.php?title=Category:German_adjectives&pagefrom=ABGESCHIEDEN%0Aabgeschieden#mw-pages" # start at this one because there is a lot of repetition in the numbers

    url = START_URL
    adjectives = set()

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
            if a and a.text: # essentially just checks that these are not empty
                # now remove multi-word adjectives
                if a and a.text and " " not in a.text:
                    if random() < ratio_to_store: # there are far too many adjectives to store all of them so we just store a random sampling of them
                        adjectives.add(a.text)

        # Find next page link
        next_link = None
        for a in mw_pages.find_all("a"):
            if a.text.lower() == "next page":
                next_link = BASE_URL + a["href"]
                break

        url = next_link

        sleep(0.001) # server curtosey - wikipedia is probably built to handle the traffic, but its a good practice nonetheless I think.

    return adjectives



unwanted_auxilaries = { "er", "sie", "Sie", "ist", "es", "sein", "sind", "der", "die", "das", "des", "den", "dem", "ein", "eine", "einen", "einer", "einem",
                        "eines", "kein", "keiner", "keinen", "keine", "keinem", "keines" }
other_unwated = set()

for unwanted in unwanted_auxilaries:
    other_unwated.add("("+unwanted+")")

unwanted_auxilaries = unwanted_auxilaries | other_unwated # union the two sets
del other_unwated



def get_forms(dictionary_form:str = "alt"):
    url = "https://en.wiktionary.org/wiki/"+dictionary_form
    forms = set()

    session = requests.Session()
    session.headers.update(HEADERS)

    res = session.get(url)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    '''
    The expandable tables of adjective forms (case, plurality and comparative/superlative/standard) are stored in <div class="NavFrame"> elements.
    The actual form are stored in <td> elements.
    '''

    page = soup.find("div", id="mw-pages") # Find the category listing

    print(page)

    nav_frames = page.find_all("div", class_="NavFrame")


    for frame in nav_frames:
        td_elements = frame.find_all("td")
        for td in td_elements:
            td_clean = td.text
            # clean the string of all articles - this is going to be annoying
            for unwanted in unwanted_auxilaries:
                if unwanted+" " in td_clean: # space after should ensure that these are their own word
                    td_clean.replace(unwanted,"")
            if td_clean != None:
                forms.add(td_clean)





if __name__ == '__main__':

    get_forms()

    '''training_percent = 0.3

    raw_adj = pd.DataFrame()
    raw_adj["form"] = []
    raw_adj["root"] = []

    base_adjectives = get_adjectives(training_percent) # get the dictionary form of the adjectives


    with open("German_Adj_Training_Data.csv", "w", newline='\n', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["form", "root"]) # add the heading
        print("Saving adjective training data:")

        for base in tqdm(base_adjectives):
            forms = get_forms(base)
            for form in forms:
                # add to the file
                    writer.writerow([form, base])
                    file.flush() # writes to file
'''

    """verbs_raw = pd.read_csv('de_verbs_raw.csv')

    from numpy.random import choice

    verbs_raw_mask = choice(len(verbs_raw), round(training_percent * len(verbs_raw)), replace=False) # choose a subset of all german verbs to train on

    verbs_raw = verbs_raw.loc[verbs_raw_mask].reset_index(drop=True)

    gen_verb_training_data_de(verbs_raw)"""