import pandas as pd
import csv
import pickle

def gen_verb_training_data_de(raw_verbs: pd.DataFrame):

    with open("German_Verb_Training_Data.csv",'w', newline='\n', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["form", "root"]) # add the heading
        for i in range(len(raw_verbs)):
            for type in {"Präsens_ich", "Präsens_du", "Präsens_er, sie, es", "Präteritum_ich", "Partizip II", "Konjunktiv II_ich", "Imperativ Singular", "Imperativ Plural"}:
                form = verbs_raw.loc[i][type]
                if form != None and form !="-" and form != "—":
                    #print(form) # debug
                    writer.writerow([form, verbs_raw.loc[i].Infinitive ]) # apparently the T5 model requires an instructional prefix, this might be why it was not working
                    file.flush() # writes to file



if __name__ == '__main__':

    verbs_raw = pd.read_csv('de_verbs_raw.csv')

    from numpy.random import choice

    verbs_raw_mask = choice(len(verbs_raw), round(0.3 * len(verbs_raw)), replace=False) # choose a subset of all german verbs to train on

    verbs_raw = verbs_raw.loc[verbs_raw_mask].reset_index(drop=True)

    gen_verb_training_data_de(verbs_raw)

    '''with open('de_verbs_raw.pickle', 'wb') as handle:
        pickle.dump(verbs_raw, handle, protocol=pickle.HIGHEST_PROTOCO

    with open('de_verbs_raw.pickle', 'rb') as handle:
        verbs_raw = pickle.load(handle)''' # pickle file is actually larger than the csv - not worth it

    #print(verbs_raw.loc[5]["Imperativ Plural"])