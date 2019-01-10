import subprocess
import os
import json

import numpy as np
from sklearn.metrics import accuracy_score
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import load_model

def x_and_y(data):
    characters, labels = [], []
    for w in data:
        sylls = []
        for syll in w['syllabified'].split('-'):
            syll = ''.join(c for c in syll if c.isalpha()).strip().lower()
            if syll:
                sylls.append(syll)
        sylls = tuple(sylls)
        stress_pattern = None

        # make sure corrections override automated analysis:
        for k in ('patterns', 'corrected_stress', 'human_annotation'):
            if w[k]:
                stress_pattern = w[k]

        if isinstance(stress_pattern, list):
            stress_pattern = tuple(stress_pattern)
        elif isinstance(stress_pattern, dict):
            stress_pattern = int(max(stress_pattern, key=stress_pattern.get)),
        else:
            stress_pattern = stress_pattern,
        
        try:
            stressed = {(len(sylls) + i) for i in stress_pattern}
        except TypeError: # in case no predictions are p-made
            stressed = {}

        chars, labs = [], []
        for syll_idx, syll in enumerate(sylls):
            for char_idx, char in enumerate(syll):
                chars.append(char)
                if char_idx == 0:
                    if syll_idx in stressed:
                        labs.append(0)
                    else:
                        labs.append(1)
                else:
                    labs.append(2)

        characters.append(chars)
        labels.append(labs)
    
    return characters, labels


def load_file(p, max_from_file=None):
    with open(p, 'r') as f:
        items = json.loads(f.read())

    if max_from_file:
        return items[:max_from_file]
    else:
        return items


def load_splits(input_dir, max_from_file=None):
    train = load_file(os.sep.join((input_dir, 'train.json')), max_from_file)
    dev = load_file(os.sep.join((input_dir, 'dev.json')), max_from_file)
    test = load_file(os.sep.join((input_dir, 'test.json')), max_from_file)

    train = x_and_y(train)
    dev = x_and_y(dev)
    test = x_and_y(test)

    return train, dev, test

def pred_to_classes(X):
    """
    * Convert the 3-dimensional representation of class labels
      (nb_words, nb_timesteps, 3) to a 2-dimensional representation
      of shape ((nb_words, nb_timesteps)).
    """
    words = []
    for w in X:
        words.append([np.argmax(p) for p in w])
    return np.array(words)

def jsonify(tokens, predictions, rm_symbols=True):
    """
    * Takes an original, unsyllabified `orig_token` (e.g. seruaes)
      and aligns it with a syllabification proposed (`segmentation`).
    * Returns the syllabified token in string format (e.g. ser-uaes).
    """
    l = []
    for token, prediction in zip(tokens, predictions):
        token = list(token)

        if rm_symbols:
            prediction = prediction[1 : len(token) + 1]
        else:
            prediction = list(prediction)
        
        syll_str, stress = '', []

        curr_syll = -1
        for p in prediction[::-1]:
            if p == 2:
                syll_str += token.pop()
            else:
                if p == 0:
                    stress.append(curr_syll)
                curr_syll -= 1
                syll_str += token.pop()
                if token:
                    syll_str += '-'
        
        syll_str = ''.join(syll_str[::-1])
        stress = stress[::-1]
        d = {'syllabified': syll_str,
             'human_annotation': stress,
             'patterns': None,
             'corrected_stress': None}
        l.append(d)
    
    return json.dumps(l, indent=4)


def load_keras_model(path, no_crf):
    custom_objects={'CRF': CRF,
                    'crf_loss': crf_loss,
                    'crf_viterbi_accuracy': crf_viterbi_accuracy}
    return load_model(path, custom_objects=custom_objects)