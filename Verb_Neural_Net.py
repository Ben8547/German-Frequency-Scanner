"""
Here we construct and train the AI model to lemmatize verbs

To accomplish this we use a ByT5 transformer model since this model looks at each charachter and thus should be better that identifying word endings.

"""

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch


########################
# Training Settings
########################

model_id = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

training_args = Seq2SeqTrainingArguments(
    output_dir="./german_lemmatizer",
    per_device_train_batch_size=8,
    num_train_epochs=5,      # How many times to loop through the data
    learning_rate=5e-5,
    predict_with_generate=True,
    fp16=True,           
)

def train_model(file:str="German_Verb_Training_Data.csv"):

    ########################
    # Load the Training Data
    ########################

    training_data = pd.read_csv(file)
    training_data = training_data.dropna(subset=['form', 'root']) # was having issues with Nones

    #training_data['form'] = "lemmatize: " + training_data['form'] # add an instruction to the training data

    dataset = Dataset.from_pandas(training_data[['form', 'root']])
    dataset = dataset.train_test_split(test_size=0.1) # Save 10% for testing

    ########################
    # Tokenize the data for training
    ########################

    def preprocess_function(examples):
        inputs = tokenizer(examples["form"], max_length=64, truncation=True, padding="max_length")
        labels = tokenizer(examples["root"], max_length=64, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    
    ########################
    # Start the trainer
    ########################

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model("./lemmatize_verb_model")
    tokenizer.save_pretrained("./lemmatize_verb_model")

def load_verb_lemmatizer(path = "./lemmatize_verb_model"):

    loaded_tokenizer = AutoTokenizer.from_pretrained(path)
    loaded_model = AutoModelForSeq2SeqLM.from_pretrained(path)

    return load_verb_lemmatizer, loaded_tokenizer

def lemmatize(verb:str, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    input_text = "lemmatize: " + verb
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # train the model
    train_model()