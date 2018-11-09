import os
import shutil
import glob

from lxml import etree
from keras.models import load_model
from stresser.vectorization import SequenceVectorizer
from stresser.modeling import build_model

def enrich_word_node(w, vectorizer, model):
    syllables = []
    for syll in w.iterfind('s'):
        if not syll.text:
            continue
        syllables.append(syll.text)

    syllables_= [s.lower().strip() for s in syllables]
    syllables_= [s for s in syllables if s]

    if len(syllables) <= 1:
        return None

    v = vectorizer.transform([syllables_])
    stresses = model.predict(v)[0].argmax(axis=-1)
    word_node = etree.Element('w')
    for sy, st in zip(syllables, stresses):
        syll_node = etree.Element('s')
        syll_node.text = sy
        syll_node.attrib['word-stress'] = str(st)
        word_node.append(syll_node)
    return word_node


def main():
    model_prefix = 'wordstresser'
    input_dir = '../syll_xml'
    output_dir = '../stress_xml'
    max_songs = None

    shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    vectorizer = SequenceVectorizer.load(model_prefix + '.vec')
    model = load_model(model_prefix + '.model')

    in_paths = glob.glob(input_dir + '/*.xml')
    for fn_idx, fn in enumerate(in_paths):
        song = etree.parse(fn)

        if max_songs and fn_idx >= max_songs:
            break


        for line_idx, line in enumerate(song.iterfind('.//l')):
            for node in line:
                if node.tag == 'w':
                    word_node = enrich_word_node(node, vectorizer, model)
                    if word_node is not None:
                        line.replace(node, word_node)
                elif node.tag == 'rhyme':
                    for wnode in node:
                        if wnode.tag == 'w':
                            word_node = enrich_word_node(wnode, vectorizer, model)
                            if word_node is not None:
                                node.replace(wnode, word_node)

        out_path = os.sep.join((output_dir, os.path.basename(fn)))
        with open(out_path, 'w') as f:
            f.write(etree.tostring(song, pretty_print=True).decode())


if __name__ == '__main__':
    main()

