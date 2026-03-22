from de_verb_lstm import lemmatize, Seq2SeqLSTM, encode
import torch as t
import pickle
import pandas as pd
from tqdm import tqdm # for progress bar

if __name__ == "__main__": # tests

    # laod the encoder
    with open("./lemmatize_adj_model/encoder.pkl", "rb") as f:
        Encoder:encode = pickle.load(f)

    # load the model
    model = Seq2SeqLSTM(len(Encoder.char2idx))
    model.load_state_dict(t.load("./lemmatize_adj_model/de_adj_lemmatizer_model.pt")) # load weights from file
    
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model.to(device)

    # lemmatize a verb

    # small test:

    for word in {"kältesten", "kranken", "krakem", "kränkeren", "ältere", "hohen", "älteren", "höheren", "hochsten"}:
        out = lemmatize(model,Encoder,word.lower(),device=device)
        print(word+' : '+out)