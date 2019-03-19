import re
import os
import argparse
import html
import shutil
import json

from tqdm import tqdm
from unidecode import unidecode
from bs4 import BeautifulSoup
import numpy as np

from keras.models import load_model
from keras_contrib.utils import save_load_utils
from stresser.vectorization import SequenceVectorizer
from stresser.modelling import build_model
import stresser.utils as u

def load_lines(fn):
    LACUNA = re.compile(r'\.\.+')

    with open(fn, 'r') as f:
        xml_text = f.read()

    xml_text = xml_text.replace('&oudpond;', '')
    xml_text = xml_text.replace('&supm;', 'm')
    xml_text = xml_text.replace('&supM;', 'm')
    xml_text = xml_text.replace('&supc;', 'c')
    xml_text = xml_text.replace('&supt;', 't')
    xml_text = xml_text.replace('&supn;', 'n')
    xml_text = xml_text.replace('&sups;', 's')
    xml_text = xml_text.replace('&supd;', 'd')
    xml_text = xml_text.replace('&supc;', 'c')
    xml_text = xml_text.replace('&uring;', 'u')
    xml_text = xml_text.replace('&lt;', '')
    xml_text = xml_text.replace('&gt;', '')
    xml_text = html.unescape(xml_text)

    soup = BeautifulSoup(xml_text, 'html.parser')
    lines = []
    for line in soup.find_all('l'):
        if line.has_attr('parse'):
            continue
        line = line.get_text().strip().lower()
        if not line or re.search(LACUNA, line):
            continue
        line = ''.join(c for c in line if c.isalpha() or c.isspace())
        line = ' '.join(line.split()).strip()
        if line:
            lines.append(line)
    return lines

def main():
    parser = argparse.ArgumentParser(description='Pollysyllable verse line stresser')
    parser.add_argument('--model_dir', type=str, default='model_s')
    parser.add_argument('--no_crf', default=False, action='store_true',
                        help='Exclude the CRF from the model')
    parser.add_argument('--infile', type=str, default='data/xml_mnl/Lutgard_K.xml')
    parser.add_argument('--outfile', type=str, default='Lutgart_poly.json')
    args = parser.parse_args()
    m_path = os.sep.join((args.model_dir, 'syllab.model'))

    vec_prefix = f'{args.model_dir}/vectorizer'
    vectorizer = SequenceVectorizer.load(vec_prefix + '.json')
    model = u.load_keras_model(m_path, no_crf=args.no_crf)    
    lines = load_lines(args.infile)
      
    syllabified = []
    for line in tqdm(lines):
        syll_words, stress_words = [], []
        words = line.split()
        if len(words) < 2:
            continue
        
        monosyllab = False
        for word in words:
            if len(word) < 2:
                monosyllab = True
                break
            
            word_v = vectorizer.transform([word])
            prediction = model.predict(word_v)
            prediction = prediction.argmax(axis=-1)[0]

            token = list(word)
            prediction = prediction[1 : len(token) + 1]
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
            sylls = syll_str.split('-')
            if len(sylls) < 2:
                monosyllab = True
                break

            int_stress = np.zeros(len(sylls), dtype=np.int32)
            for idx in stress[::-1]:
                int_stress[idx] = 1
            int_stress = [int(i) for i in int_stress]
            
            syll_words.append(sylls)
            stress_words.append(int_stress)
        
        if not monosyllab:
            syllabified.append({'syllables': syll_words,
                                'stresses': stress_words})
    print(f'-> {len(syllabified)} purely polysyllabic verses')
    with open(args.outfile, 'w') as f:
        f.write(json.dumps(syllabified, indent=4))

if __name__ == '__main__':
    main()