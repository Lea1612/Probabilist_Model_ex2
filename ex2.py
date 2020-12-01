import sys
from collections import Counter
import math


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

def write_output(output_filename, output):
    with open(output_filename,"w") as f:
        for i, row in enumerate(output):
            f.write(f"#Output{i}\t{row}\n")
    print("\n".join([str(i) for i in output]))


class LidstoneModel(object):
    def __init__(self, train, val, test):
        self.training_event_occuration = Counter(train)
        self.validation_event_occuration = Counter(val)
        self.validation = val
        self.test = test
        self.training_set_size = len(train)
        self.validation_set_size = len(val)
        self.test_set_size = len(test)
        self.different_events_counter_train = len(self.training_event_occuration)

    def calculate_prob(self, word=None):
        word_count = self.training_event_occuration[word]
        return word_count / self.training_set_size

    def calculate_prob_lambda(self, word=None, lambda_var=0.1):
        word_count = self.training_event_occuration[word]
        return float(word_count + lambda_var) / \
               (self.training_set_size + lambda_var * vocabulary_size)

    # def calculate_perplexity(self, lambda_var=0.01):
    #     sigma = sum([math.log2(self.calculate_prob_lambda(w, lambda_var)) for w in self.validation])
    #     return 2**((-1 / self.validation_set_size) * sigma)

    def lidstone_perplexity(self, dataset, dataset_size, lambda_var=0.01):
        sigma = sum([math.log2(self.calculate_prob_lambda(w, lambda_var)) for w in dataset])
        return 2**((-1 / dataset_size) * sigma)


    def best_lambda(self):
        perplexity_values = []
        
        for lamda in range(1,201):
            perplexity_values.append((self.lidstone_perplexity(self.validation, self.validation_set_size, lambda_var= lamda / 100), lamda / 100))

        min_preplexity, min_lambda = min(perplexity_values)

        return min_preplexity, min_lambda


class HeldOut(object):

    def __init__(self, ho_train, heldout):
        self.ho_train = ho_train
        self.heldout = heldout
        self.ho_training_occuration = Counter(ho_train)
        self.ho_heldout_occuration = Counter(heldout)
        self.ho_training_size = len(ho_train)
        self.ho_heldout_size = len(heldout)
        
    def heldout_proba(self, word = None) :
        same_frequence_words_train = []
        word_occuration_train = self.ho_training_occuration[word]

        if word_occuration_train > 0:
            for w, count in self.ho_training_occuration.items():
                if count == word_occuration_train:
                    same_frequence_words_train.append(w)

            N_r = len(same_frequence_words_train)


        else: 
            same_frequence_words_train = [word for word in self.heldout if word not in self.ho_train]
            N_r = vocabulary_size - self.ho_training_size


        T_r = sum([self.ho_heldout_occuration[word] for word in same_frequence_words_train])
        f_emp = T_r / N_r

        return f_emp / self.ho_heldout_size


def main():
    output = [None]*29
    dev_set_file_name = "dataset/develop.txt"   # sys.argv[1]
    test_set_file_name = "dataset/test.txt"  # sys.argv[2]
    input_word = "honduras"  # sys.argv[3]
    output_filename = "output.txt"  # sys.argv[4]

    train_words = string_words(dev_set_file_name)   # Sequence of events from the dev set
    test_words = string_words(test_set_file_name)

### 1. Init ###

    output[1] = dev_set_file_name
    output[2] = test_set_file_name
    output[3] = input_word
    output[4] = output_filename
    output[5] = vocabulary_size
    output[6] = 1 / vocabulary_size  # Uniform proba. All events have same proba to occur.

### 2. Dvlpt set prepocessing ###

    output[7] = len(train_words) 

### 3. Lidstone model training ###

    threshold = round(0.9 * len(train_words))
    training_set = train_words[:threshold]  # The first 90% to train
    validation_set = train_words[threshold:]  # The last 10% to validation
    lidstone_model = LidstoneModel(training_set, validation_set, test_words)

    output[8] = lidstone_model.validation_set_size     # Nb of events in validation
    output[9] = lidstone_model.training_set_size       # Nb of events in training set
    output[10] = lidstone_model.different_events_counter_train # Nb of different events in train
    output[11] = lidstone_model.training_event_occuration[input_word]  # Nb of times the input_word occurs in train
    output[12] = lidstone_model.calculate_prob(input_word) # MLE train
    output[13] = lidstone_model.calculate_prob() #MLE on unseen words. word = None 
    output[14] = lidstone_model.calculate_prob_lambda(word=input_word) # Proba Lidstone with lambda = 0.1
    output[15] = lidstone_model.calculate_prob_lambda() # On unseen words. word = None
    
    lambda_var = [0.01, 0.1, 1.0]

    
    output[16] = lidstone_model.lidstone_perplexity(lidstone_model.validation, lidstone_model.validation_set_size, lambda_var[0]) # Perplexity on valid. lambda = 0.01
    output[17] = lidstone_model.lidstone_perplexity(lidstone_model.validation, lidstone_model.validation_set_size, lambda_var[1]) # lambda = 0.1
    output[18] = lidstone_model.lidstone_perplexity(lidstone_model.validation, lidstone_model.validation_set_size, lambda_var[2]) # lambda = 1

    # output[20] = min(output[16:19]) # minimized perplexity on valid
    # output[19] = lambda_var[output[16:19].index(output[20])] # lambda that minimizes perplexity on valid
    
    output[19] = lidstone_model.best_lambda()[1] # lambda that minimizes perplexity on valid
    output[20] = lidstone_model.best_lambda()[0] # minimized perplexity on valid

    ### 4.  Held out training ###

    threshold_heldout = round(0.5 * len(train_words)) # Half for train and half for heldout
    heldout_training = train_words[:threshold_heldout]
    heldout = train_words[threshold_heldout:]

    held_out = HeldOut(heldout_training, heldout)

    output[21] = held_out.ho_training_size # Nb of events in train
    output[22] = held_out.ho_heldout_size # Nb of events in heldout

    output[23] = held_out.heldout_proba(word = input_word) # Heldout on input word
    output[24] = held_out.heldout_proba() # Pb for word = None ???

    ### Debug ###

    ### 5. Models evaluation on test set ###
    
    output[25] = len(test_words) # Nb of events in test

    output[26] = lidstone_model.lidstone_perplexity(lidstone_model.test, lidstone_model.test_set_size, output[19])

    print('he')
    write_output(output_filename, output)



if __name__ == "__main__":
    main()
