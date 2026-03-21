import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle # for saving the encoder used with the trainer

######
# Build the "vocabulary" that the model will use to encode strings
######

class encode:

    def __init__(self, training_path : str = "German_Verb_Training_Data.csv"):
        self.training_path = training_path
        self.build_vocab()

    def build_vocab(self): # Collect all characters from your dataset

        self.training_data = pd.read_csv(self.training_path)

        self.all_chars = set()

        for form, root in zip(self.training_data["form"], self.training_data["root"]):
            self.all_chars.update(form) # add to the set
            self.all_chars.update(root)

        self.PAD = "<pad>" # Add special tokens used in the encoding stage
        self.SOS = "<sos>"
        self.EOS = "<eos>"

        self.chars = [self.PAD, self.SOS, self.EOS] + sorted(list(self.all_chars))

        self.char2idx = {c: i for i, c in enumerate(self.chars)} # assign an interger to each character
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def encode(self, word, max_len=40):
        seq = [self.char2idx[self.SOS]] + [self.char2idx[c] for c in word] + [self.char2idx[self.EOS]]

        if len(seq) > max_len:
            seq = seq[:max_len-1] + [self.char2idx[self.EOS]]

        seq += [self.char2idx[self.PAD]] * (max_len - len(seq))
        return seq[:max_len]
    


class VerbDataset(Dataset):
    def __init__(self, encoder:encode):
        forms, roots = (encoder.training_data["form"], encoder.training_data["root"])
        self.X = [encoder.encode(f) for f in forms]
        self.y = [encoder.encode(r) for r in roots]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)
    

class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt):
        # src: (batch, seq_len)
        # tgt: (batch, seq_len)

        embedded_src = self.embedding(src)
        _, (hidden, cell) = self.encoder(embedded_src)

        embedded_tgt = self.embedding(tgt)
        outputs, _ = self.decoder(embedded_tgt, (hidden, cell))

        logits = self.fc(outputs)
        return logits
    


def lemmatize(model, encoder:encode, word, max_len=40, device:str="cpu"):
    #model.to(device)
    #model.eval()

    src = torch.tensor([encoder.encode(word)]).to(device)

    with torch.no_grad():
        embedded = model.embedding(src)
        _, (hidden, cell) = model.encoder(embedded)

        tgt = torch.tensor([[encoder.char2idx[encoder.SOS]]]).to(device)

        result = []

        for _ in range(max_len):
            embedded_tgt = model.embedding(tgt)
            output, (hidden, cell) = model.decoder(embedded_tgt, (hidden, cell))
            logits = model.fc(output[:, -1])

            pred = logits.argmax(dim=-1).item()

            if pred == encoder.char2idx[encoder.EOS]:
                break

            result.append(encoder.idx2char[pred])

            tgt = torch.tensor([[pred]]).to(device)

    return "".join(result)
    

if __name__ == "__main__": # train the model

    epochs = 10 #number of epochs to run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Encode = encode() # essentially just loads in the data and assigns each character and integer that LSTM can interpret.

    model = Seq2SeqLSTM(len(Encode.char2idx))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=Encode.char2idx[Encode.PAD])

    dataset = VerbDataset(Encode)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    model.train() # set to training mode

    for epoch in range(epochs):
        total_loss = 0

        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()

            # Shift target for teacher forcing
            output = model(src, tgt[:, :-1])

            loss = criterion(
                output.reshape(-1, len(Encode.char2idx)),
                tgt[:, 1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "./lemmatize_verb_model/de_verb_lemmatizer_model.pt")

    with open("./lemmatize_verb_model/encoder.pkl", "wb") as f: # save the encoder or else we can't ever use the model accurately
        pickle.dump(Encode, f)