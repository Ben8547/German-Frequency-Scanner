import csv

with open("part_of_speech_training_data.csv",'w', encoding="UTF-8", newline='\n') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(['word','part of speech'])

    with open("adj_de_training_data.csv",'r', encoding="UTF-8", newline='\n') as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            if row[0]!='form':
                writer.writerow([row[0],"verb"])

    with open("German_Verb_Training_Data.csv",'r', encoding="UTF-8", newline='\n') as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            if row[0]!='form':
                writer.writerow([row[0],"adjective"])

    
            