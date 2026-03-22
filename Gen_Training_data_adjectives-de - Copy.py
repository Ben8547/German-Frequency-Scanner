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



unwanted_auxilaries = [ "sein", "sind", "der", "die", "das", "des", "den", "dem", "ein", "eine", "einen", "einer", "einem",
                        "eines", "kein", "keiner", "keinen", "keine", "keinem", "keines", "er", "sie", "Sie", "ist", "es", "am"] # order matters so I can't use a set (es cannot go before des)
other_unwated = list()

for unwanted in unwanted_auxilaries:
    other_unwated.append("("+unwanted+")")

unwanted_auxilaries = unwanted_auxilaries + other_unwated # union the two sets
del other_unwated

def extract_de_fragment(soup_object:BeautifulSoup) -> BeautifulSoup :
    #out = soup_object.find("h2", id = "German") # this just extracts the title - but it does work for that - will probably need to use RE to get what I want
    #print(str(soup_object)) # debug
    out = re.search("<h2 id=\"German\">German</h2>(.*)",str(soup_object),re.DOTALL) # re.DOTALL allows .* to include line breaks hence why it was breaking before
    out = BeautifulSoup(out.group(1), "html.parser") # return to soup
    return out


def get_forms(dictionary_form:str = "alt"):
    url = "https://en.wiktionary.org/wiki/"+dictionary_form
    forms = set()

    session = requests.Session()
    session.headers.update(HEADERS)

    res = session.get(url)
    res.raise_for_status() # seems to raise an error if an error was encountered when requesting the page - should probably include for safety

    soup = BeautifulSoup(res.text, "html.parser")

    soup = extract_de_fragment(soup) # extract only the german words

    '''
    The expandable tables of adjective forms (case, plurality and comparative/superlative/standard) are stored in <div class="NavFrame"> elements.
    The actual form are stored in <td> elements.
    '''

    page = soup.find_all("div", class_="NavContent") # Find the category listing

    for frame in page[0:3]: # was getting extraneuous values when more than three were allowe
        td_elements = frame.find_all("td")
        for td in td_elements:
            td_clean = td.text
            # clean the string of all articles - this is going to be annoying
            for unwanted in unwanted_auxilaries:
                if unwanted+' ' in td_clean: # space after should ensure that these are their own word
                    td_clean = td_clean.replace(unwanted,"")
            td_clean = td_clean.replace("\n","")
            td_clean = td_clean.replace("—","")
            td_clean = td_clean.replace(" ","")
            if td_clean != "":
                forms.add(td_clean)

    return forms





if __name__ == '__main__':

    #forms = get_forms("hochwertig") # test

    with open("adj_de_training_data.csv", "w", encoding="UTF-8") as train_data: # reset the training data
        writer = csv.writer(train_data)
        writer.writerow(["form","root"])
    
    # generate the training data
    with open("500_common_german_adjectives.csv", "r", encoding="UTF-8") as adjective_list:
        reader = csv.reader(adjective_list)
        for adj in reader:
            forms = get_forms(adj[0])
            with open("adj_de_training_data.csv", "a", encoding="UTF-8",newline='\n') as train_data: # use append mode for this so that we don't overwrite the file
                writer = csv.writer(train_data)
                for form in forms:
                    writer.writerow([form,adj[0]])



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