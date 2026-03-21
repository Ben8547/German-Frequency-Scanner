from de_verb_lstm import lemmatize, Seq2SeqLSTM, encode
import torch as t
import pickle
import pandas as pd

if __name__ == "__main__": # tests

    # laod the encoder
    with open("./lemmatize_verb_model/encoder.pkl", "rb") as f:
        Encoder:encode = pickle.load(f)

    # load the model
    model = Seq2SeqLSTM(len(Encoder.char2idx))
    model.load_state_dict(t.load("./lemmatize_verb_model/de_verb_lemmatizer_model.pt")) # load weights from file
    
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model.to(device)

    # lemmatize a verb

    # small test:

    for word in {"ausgekommen", "Sein", "haben", "gegessen", "gegangen", "liest", "laden ein", "kommt aus"}:
        out = lemmatize(model,Encoder,word.lower(),device=device)
        print(word+' : '+out)


    # large test: 

    all_verbs = pd.read_csv("Verb_Benchmarking.csv")  

    total_verbs = len(all_verbs)

    correct = 0

    for i in range(total_verbs):
        word = all_verbs["form"][i]
        correct_answer = all_verbs["root"][i]
        answer = lemmatize(model,Encoder,word.lower(),device=device)
        if answer == correct_answer:
            correct += 1

    print(f"Fidelity of model is {correct/total_verbs*100.}%")