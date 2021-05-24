# Lea Setruk  Vladimir Balagula  

import sys
from collections import Counter
import math
from datetime import datetime

vocabulary_size = 300000

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

# Write the output with the right format
def write_output(output_filename, output):
    with open(output_filename, "w") as f:
        f.write(f"{'#Students'}\t{'Lea Fanny Setruk'}\t{'Vladimir Balagula'}\t{'345226179'}\t{'323792770'}\n")
        for i, row in enumerate(output):
            f.write(f"#Output{i+1}\t{row}\n")
    print("\n".join([str(i) for i in output if i]))


# Lidstone model on train and on validation sets.
class LidstoneModel(object):

    def __init__(self, train, val=None, test=None):
        self.training_event_occuration = Counter(train)
        self.test = test
        self.training_set_size = len(train)
        self.different_events_counter_train = len(self.training_event_occuration)
        if val:
            self.validation_event_occuration = Counter(val)
            self.validation_set_size = len(val)
            self.validation = val
        if test:
            self.test_set_size = len(test)


    def calculate_prob(self, word='unseen_word'):
        word_count = self.training_event_occuration[word]
        return float(word_count) / float(self.training_set_size)

    def calculate_prob_lambda(self, word='unseen_word', lambda_var=0.1, frequency=None):
        word_count = self.training_event_occuration[word] if word else frequency
        return float(word_count + lambda_var) / \
               (self.training_set_size + lambda_var * vocabulary_size)

    def lidstone_perplexity(self, dataset, dataset_size, lambda_var=0.01):
        sigma = sum([math.log2(self.calculate_prob_lambda(w, lambda_var)) for w in dataset])
        return 2**((-1 / dataset_size) * sigma)

    def best_lambda(self):
        perplexity_values = []
        
        for lamda in range(1,201):
            perplexity_values.append((self.lidstone_perplexity(self.validation, self.validation_set_size, lambda_var= lamda / 100), lamda / 100))

        min_preplexity, min_lambda = min(perplexity_values)

        return min_preplexity, min_lambda


# Held Out model with the divison of the data set in 2 halves. One to train and one to held out.
class HeldOut(object):

    def __init__(self, ho_train, heldout, test_words=None):
        self.ho_train = set(ho_train)
        self.heldout = set(heldout)
        self.ho_training_occuration = Counter(ho_train)

        self.ho_heldout_occuration = Counter(heldout)

        self.ho_training_size = len(ho_train)
        self.ho_heldout_size = len(heldout)
        if test_words:
            self.test = test_words
            self.test_size = len(test_words)

    def calculate_N_r(self, word, frequency):
        same_frequence_words_train = []
        word_occuration_train = self.ho_training_occuration[word] if word else frequency

        if word_occuration_train > 0:
            for w, count in self.ho_training_occuration.items():
                if count == word_occuration_train:
                    same_frequence_words_train.append(w)

            N_r = len(same_frequence_words_train)

        else:
            same_frequence_words_train = [w for w in self.heldout if w not in self.ho_train]
            N_r = vocabulary_size - len(set(self.ho_train))

        return N_r, same_frequence_words_train

    def heldout_proba(self, word='unseen_word',frequency=None):
        N_r, same_frequence_words_train = self.calculate_N_r(word, frequency)

        T_r = sum([self.ho_heldout_occuration[word] for word in same_frequence_words_train])
        f_emp = T_r / N_r

        return f_emp / self.ho_heldout_size

    def heldout_perplexity(self, dataset, dataset_size):
        sigma = sum([math.log2(self.heldout_proba(w)) for w in dataset])
        return 2**((-1 / dataset_size) * sigma)


# Last question. Need to write a table with the values of f_lambda (for lidstone)
# f_h for heldout, N_r the nb of words with same frequence, t_r like in the exercice
def create_table(lidstone_model, held_out, output):
    matrix = []

    for i in range(10):

        f_lambda = lidstone_model.calculate_prob_lambda(word=None,
                                                        lambda_var=output[18],
                                                        frequency=i)*lidstone_model.training_set_size

        f_h = held_out.heldout_proba(word=None, frequency=i) * held_out.ho_training_size

        N_r, same_frequence_words_train = held_out.calculate_N_r(word=None, frequency=i)

        t_r = sum([held_out.ho_heldout_occuration[word] for word in same_frequence_words_train])

        matrix.append(f"{i}\t{round(f_lambda, 5)}\t{round(f_h, 5)}\t{round(N_r, 5)}\t{round(t_r,5)}")

    return '\n' + "\n".join(matrix)


def main():
    output = []
    dev_set_file_name = sys.argv[1]
    test_set_file_name = sys.argv[2]
    input_word = sys.argv[3]
    output_filename = sys.argv[4]

    train_words = string_words(dev_set_file_name)   # Sequence of events from the dev set
    test_words = string_words(test_set_file_name)

    ### 1. Init ###

    output.append(dev_set_file_name)
    output.append(test_set_file_name)
    output.append(input_word)
    output.append(output_filename)
    output.append(vocabulary_size)
    output.append(1 / vocabulary_size)  # Uniform proba. All events have same proba to occur.

    ### 2. Dvlpt set prepocessing ###

    output.append(len(train_words))

    ### 3. Lidstone model training ###

    threshold = round(0.9 * len(train_words))
    training_set = train_words[:threshold]  # The first 90% to train
    validation_set = train_words[threshold:]  # The last 10% to validation
    lidstone_model = LidstoneModel(training_set, validation_set, test_words)

    output.append(lidstone_model.validation_set_size)     # Nb of events in validation
    output.append(lidstone_model.training_set_size)       # Nb of events in training set
    output.append(lidstone_model.different_events_counter_train) # Nb of different events in train
    output.append(lidstone_model.training_event_occuration[input_word])  # Nb of times the input_word occurs in train
    output.append(lidstone_model.calculate_prob(input_word)) # MLE train
    output.append(lidstone_model.calculate_prob()) #MLE on unseen words. word = None
    output.append(lidstone_model.calculate_prob_lambda(word=input_word)) # Proba Lidstone with lambda = 0.1
    output.append(lidstone_model.calculate_prob_lambda()) # On unseen words. word = None
    
    lambda_var = [0.01, 0.1, 1.0]

    output.append(lidstone_model.lidstone_perplexity(lidstone_model.validation, lidstone_model.validation_set_size, lambda_var[0])) # Perplexity on valid. lambda = 0.01
    output.append(lidstone_model.lidstone_perplexity(lidstone_model.validation, lidstone_model.validation_set_size, lambda_var[1])) # lambda = 0.1
    output.append(lidstone_model.lidstone_perplexity(lidstone_model.validation, lidstone_model.validation_set_size, lambda_var[2])) # lambda = 1
 
    output.append(lidstone_model.best_lambda()[1]) # lambda that minimizes perplexity on valid
    output.append(lidstone_model.best_lambda()[0]) # minimized perplexity on valid

    ### 4.  Held out training ###

    threshold_heldout = round(0.5 * len(train_words)) # Half for train and half for heldout
    heldout_training = train_words[:threshold_heldout]
    heldout = train_words[threshold_heldout:]

    held_out = HeldOut(heldout_training, heldout, test_words)

    output.append(held_out.ho_training_size) # Nb of events in train
    output.append(held_out.ho_heldout_size) # Nb of events in heldout

    output.append(held_out.heldout_proba(word=input_word)) # Heldout on input word
    output.append(held_out.heldout_proba())  # Pb for unseen words

    ### Debug ###

    lidstone_check = LidstoneModel(train_words)
    prob = lidstone_check.calculate_prob()*(vocabulary_size - len(lidstone_check.training_event_occuration))
    for word in lidstone_check.training_event_occuration.keys():
        prob += lidstone_check.calculate_prob(word)
    if round(prob, 5) == 1.0:
        print("Lidstone test is pass")
    else:
        print("Lidstone out test failed")
    held_out_check = HeldOut(train_words, ['xxxx', 'yyy', 'zzzz'])
    prob = held_out_check.heldout_proba()*(vocabulary_size - len(held_out_check.ho_train))
    for word in held_out_check.ho_training_occuration.keys():
        prob += held_out_check.heldout_proba(word=word)
    if round(prob, 5) == 1.0:
        print("Held out all unknown words is pass")
    else:
        print("Held out test failed")

    held_out_check = HeldOut(train_words, ['xxxx', 'unsworth', 'for'])
    prob = held_out_check.heldout_proba() * (vocabulary_size - len(held_out_check.ho_train))
    for word in held_out_check.ho_training_occuration.keys():
        prob += held_out_check.heldout_proba(word=word)
    if round(prob, 5) == 1.0:
        print("Held out test some words are know is pass")
    else:
        print("Held out test failed")

    ### 5. Models evaluation on test set ###
    output.append(len(test_words))  # Nb of events in test
 
    output.append(lidstone_model.lidstone_perplexity(lidstone_model.test, lidstone_model.test_set_size, output[18]))
    output.append(held_out.heldout_perplexity(held_out.test, held_out.test_size))

    output.append("L" if output[25] < output[26] else "H")

    ### Table ###

    output.append(create_table(lidstone_model, held_out, output))

    write_output(output_filename, output)


if __name__ == "__main__":
    main()
