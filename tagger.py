import numpy as np
from hmm import HMM

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    idx = 0
    for data in unique_words:
        if data not in word2idx.keys():
            word2idx[data] = idx
        idx+=1
                
    idx = 0
    for tag_id in range(S):
        if tags[tag_id] not in tag2idx.keys():
            tag2idx[tags[tag_id]] = idx
        idx+=1

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################
    
    word_count = dict()
    tag_transition_count = dict()
    total_transition_count = dict()
    tag_count = dict()
    tag_observation_count = dict()
    first_tag_count = dict()
    
    for line in train_data:
        current_words = line.words
        current_tags = line.tags
        
        if current_tags[0] not in first_tag_count.keys():
            first_tag_count[current_tags[0]] = 1
        else:
            first_tag_count[current_tags[0]] += 1

        for count, word in enumerate(current_words):
            present_tag = current_tags[count]
            if word not in word_count.keys():
                word_count[word] = 1
            else:
                word_count[word] += 1

            if current_tags[count] not in tag_count.keys():
                tag_count[current_tags[count]] = 1
            else:
                tag_count[current_tags[count]] += 1
            
            if(not(count == 0)): 
               if(str(last_tag) not in total_transition_count.keys()):
                    total_transition_count[str(last_tag)] = 1
               else:
                    total_transition_count[str(last_tag)] += 1

            if((str(present_tag), word) not in tag_observation_count.keys()):
                tag_observation_count[(str(present_tag),word)] = 1
            else:
                tag_observation_count[(str(present_tag),word)] +=1

            if(not(count == 0)):
                if((str(last_tag), str(present_tag)) not in tag_transition_count.keys()):
                    tag_transition_count[(str(last_tag),str(present_tag))] = 1
                else:
                    tag_transition_count[(str(last_tag),str(present_tag))] +=1
           
            last_tag = present_tag
        
    #Compute Pi, A and B
    for tag in tags:
        tag_id = tag2idx[tag]
        if(tag not in first_tag_count.keys()):
            pi[tag_id] = 0
        else:    
            pi[tag_id] = first_tag_count[tag]/len(train_data)

        for Observation in tag_observation_count:
            if((tag_count[tag] is not 0) and not((str(tag), Observation[1]) not in tag_observation_count.keys())):
                B[tag_id][word2idx[Observation[1]]] = tag_observation_count[str(tag), Observation[1]]/tag_count[tag]
            else:
                B[tag_id][word2idx[Observation[1]]] = 0
    
        for tag2 in tags:
            if((total_transition_count[tag] is not 0) and not((str(tag), str(tag2)) not in tag_transition_count.keys())):
                A[tag_id][tag2idx[tag2]] = tag_transition_count[str(tag), str(tag2)]/total_transition_count[str(tag)]
            else:
                A[tag_id][tag2idx[tag2]] = 0
    
    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    index = max(model.obs_dict.values()) + 1
    prob = 10**-6
    for line in test_data:
        for count,word in enumerate(line.words):
            if word not in model.obs_dict.keys():
                model.obs_dict[word] = index
                model.B = np.append(model.B, np.ones([len(tags),1])*prob, axis =1)
                index += 1
        tagging.append(model.viterbi(line.words))

    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
