"""
First, you need to compile the Bouma baseline:
>>> gcc -o hyphen hyphenate_mnl.c
"""

import subprocess
import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import Levenshtein

import stresser.utils as u

def main():

    def eval(silver_file, gold_file):
        silver = u.load_file(silver_file)
        _, silver_y = u.x_and_y(silver)

        gold = u.load_file(gold_file)
        _, gold_y = u.x_and_y(gold)
        
        acc_syll = accuracy_score([i for s in gold_y for i in s],
                                [i for s in silver_y for i in s])
        f1_syll = f1_score([i for s in gold_y for i in s],
                           [i for s in silver_y for i in s],
                           average='macro')
        acc_token = accuracy_score([str(s) for s in gold_y], 
                                   [str(s) for s in silver_y])

        lev = np.mean([Levenshtein.distance(''.join([str(_) for _ in g]), \
                                            ''.join([str(_) for _ in s])) \
                       for g, s in zip(gold_y, silver_y)])
            
        return acc_syll, f1_syll, acc_token, lev

    print('Plain CRF baseline:')
    acc_syll, f1_syll, acc_token, lev = eval('model_b/silver_dev.json', 'data/splits/dev.json')
    print('- dev scores:')
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)
    print('   - Levenshtein (token):', lev)

    acc_syll, f1_syll, acc_token, lev = eval('model_b/silver_test.json', 'data/splits/test.json')
    print('- test scores:')
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)
    print('   - Levenshtein (token):', lev)

    print('Our system (LSTM + CRF):')
    acc_syll, f1_syll, acc_token, lev = eval('model_s/silver_dev.json', 'data/splits/dev.json')
    print('- dev scores:')
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)
    print('   - Levenshtein (token):', lev)

    acc_syll, f1_syll, acc_token, lev = eval('model_s/silver_test.json', 'data/splits/test.json')
    print('- test scores:')
    print('   - acc (char):', acc_syll)
    print('   - f1 (char):', f1_syll)
    print('   - acc (token):', acc_token)
    print('   - Levenshtein (token):', lev)
    
if __name__ == '__main__':
    main()