import sys
from collections import Counter

# Read the file, only the sentences and not the lines with topics that start with <train or <set
# return a string of words (a sequence of words)
def string_words(file_name):
    words_sequence = []

    with open(file_name, 'r') as f:
        for line in f:
            if "<TRAIN" not in line and "<TEST" not in line and line != "\n":
                for word in line.split():
                    words_sequence.append(word)

    return words_sequence


def main():
    output = [i for i in range(1,29)]
    dev_set_file_name = "dataset/develop.txt" # sys.argv[1]
    test_set_file_name = "dataset/develop.txt" # sys.argv[2]
    input_word = "word" # sys.argv[3]
    output_filename = "output.txt" # sys.argv[4]

    vocabulary_size = 300000

    train_words = string_words(dev_set_file_name) #Sequence of events from the dev set
    test_words = string_words(test_set_file_name)

### 1. Init ###

    output[1] = dev_set_file_name
    output[2] = test_set_file_name
    output[3] = input_word
    output[4] = output_filename
    output[5] = vocabulary_size
    output[6] = 1 / vocabulary_size #Uniform proba. All events have same proba to occur.

### 2. Dvlpt set prepocessing ###

    output[7] = len(train_words) 

### Lidstone model training ###

    threshold = round(0.9 * len(train_words))

    training_set = train_words[:threshold] # The first 90% to train
    validation_set = train_words[threshold:] #The last 10% to validation

    training_event_occuration = Counter(training_set)
    validation_event_occuration = Counter(validation_set)

    training_set_size = len(training_set)   
    validation_set_size = len(validation_set)
    different_events_counter_train = len(training_event_occuration)

    output[8] = validation_set_size     # Nb of events in validation 
    output[9] = training_set_size       # Nb of events in training set
    output[10] = different_events_counter_train # Nb of different events in train
    output[11] = training_event_occuration[input_word] # Nb of times the input_word occurs in train
    output[12] = training_event_occuration[input_word] / training_set_size #MLE, no smoothing
    # output[13] = training_event_occuration['unseen-word'] / training_set_size #MLE on unseen words

    print(output)

if __name__ == "__main__":
    main()