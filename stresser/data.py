import json
import numpy as np

def load_word_stresses(fn):
    characters, labels = [], []

    data = json.load(open(fn, 'r'))
    for w in data:
        sylls = []
        for syll in data[w]['syllabified'].split('-'):
            syll = ''.join(c for c in syll if c.isalpha()).strip().lower()
            if syll:
                sylls.append(syll)
        sylls = tuple(sylls)

        stress_pattern = None
        for k in ('patterns', 'corrected_stress', 'human_annotation'):
            if data[w][k]:
                stress_pattern = data[w][k]

        if isinstance(stress_pattern, list):
            stress_pattern = tuple(stress_pattern)
        elif isinstance(stress_pattern, dict):
            stress_pattern = int(max(stress_pattern, key=stress_pattern.get)),
        else:
            stress_pattern = stress_pattern,

        stressed = {(len(sylls) + i) for i in stress_pattern}

        chars, labs = [], []
        for syll_idx, syll in enumerate(sylls):
            for char_idx, char in enumerate(syll):
                chars.append(char)
                if char_idx == 0:
                    if syll_idx in stressed:
                        labs.append(1)
                    else:
                        labs.append(2)
                else:
                    labs.append(3)

        characters.append(chars)
        labels.append(labs)

    return characters, labels
