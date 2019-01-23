import re
import os
import argparse
import html
import shutil
import json

from unidecode import unidecode
from bs4 import BeautifulSoup

from keras.models import load_model
from keras_contrib.utils import save_load_utils
from stresser.vectorization import SequenceVectorizer
from stresser.modelling import build_model
import stresser.utils as u


def main():

    parser = argparse.ArgumentParser(description='Pollysyllable verse line stresser')
    
    parser.add_argument('--model_dir', type=str,
                        default='model_s')
    parser.add_argument('--no_crf', default=False, action='store_true',
                        help='Exclude the CRF from the model')

    args = parser.parse_args()

    m_path = os.sep.join((args.model_dir, 'syllab.model'))

    vec_prefix = 'model_s/vectorizer'
    input_dir = 'data/xml_mnl'
    output_dir = 'stressed_file'


    try:
        shutil.rmtree(output_dir)
    except:
        pass

    os.mkdir(output_dir)

## LOAD MODELS ##
    
    vectorizer = SequenceVectorizer.load(vec_prefix + '.json')
    model = u.load_keras_model(m_path, no_crf=args.no_crf)
    print("MODEL LOADED")
    
## CLEANING XML DATA ##

    LACUNA = re.compile(r'\.\.+')

    for entry in os.scandir(input_dir):
        if not entry.path.endswith(".xml"):
            continue
        print('---> parsing:', entry.path)

        with open(entry.path, 'r') as f:
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
            text = line.get_text().strip()
            if (not text) or (re.search(LACUNA, text)):
                continue
            else:
                lines.append(text)
        
        clean_lines = []
        for line in lines:
            line = line.lower()
            clean_line = ''
            for char in line:
                if char.isalpha() or char.isspace():
                    clean_line += char
            clean_line = clean_line.strip()
            if clean_line:
                clean_lines.append(clean_line)

        raw_lines = [line.split() for line in clean_lines]

## SYLLABIFYING AND STRESSING VERSE LINES ##
        
        syllabified_lut = []
        for line in raw_lines:
            syll_line, stress_line = [], []
            for word in line:
                word_v = vectorizer.transform([word]) 
                predictions = model.predict(word_v)
                predictions = predictions.argmax(axis=-1)[0]

                sylls, stress, curr_syll = [], [], []

                for char, label in zip(word, predictions):
                    if label == 3:
                        curr_syll.append(char)
                    else:
                        if label == 1:
                            stress.append(1)
                        elif label == 2:
                            stress.append(0)

                        if curr_syll:
                            sylls.append(curr_syll)
                            curr_syll = []
                        curr_syll.append(char)

                if curr_syll:
                    sylls.append(curr_syll)

                sylls = [''.join(sy) for sy in sylls]

                syll_line.append(sylls)
                stress_line.append(stress)

## APPENDING RESULTS TO LIST OF ONLY POLLYSYLLABLE VERSE LINES ##
            include = True
            for word in syll_line:
                if len(word) == 1:
                    include = False
            if include:
                syllabified_lut.append(syll_line)
                print(syll_line)
                print(stress_line)

if __name__ == '__main__':
    main()