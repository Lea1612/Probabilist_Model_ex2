import sys
from collections import Counter

def string_words(file_name):
    words_sequence = []

    with open(file_name) as f:
        for line in f:
            # if "<TRAIN" not in line and "<TEST" not in line and line != "\n":
                [words_sequence.append(word) for word in line.split()]
    f.close()

    return words_sequence

def lidstone(word_frequency, lamda, training_set_size, vocabulary_size):
    # The formula to calculate the Lidstone smoothing according to the lectures.
    return (word_frequency + lamda) / (training_set_size + lamda * vocabulary_size)

def main():
    # output = dict([(key, None) for key in range(1, 29)])
    output = [i for i in range(1,29)]
    dev_set_file_name = "dataset/develop.txt" # sys.argv[1]
    test_set_file_name = "dataset/develop.txt" # sys.argv[2]
    input_word = "word" # sys.argv[3]
    output_filename = "output.txt" # sys.argv[4]

    # Given in the PDF File, the vocabulary is 300,000 length.
    vocabulary_size = 300000

    train_words = string_words(dev_set_file_name)
    test_words = string_words(test_set_file_name)

    output[1] = dev_set_file_name
    output[2] = test_set_file_name
    output[3] = input_word.lower()
    output[4] = output_filename
    output[5] = vocabulary_size

    threshold = round(0.9 * len(train_words))
    training_set = train_words[:threshold]
    validation_set = train_words[threshold:]

    training_event_occuration = Counter(training_set)
    validation_event_occuration = Counter(validation_set)

    training_set_size = len(training_set)

    output[8] = len(validation_set)
    output[9] = training_set_size
    output[10] = len(training_event_occuration)
    output[11] = training_event_occuration[input_word]
    output[12] = training_event_occuration[input_word] / training_set_size
    output[13] = training_event_occuration['unseen-word'] / training_set_size

    print(output)

if __name__ == "__main__":
    main()