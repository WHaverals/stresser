import argparse
import numpy as np

import Levenshtein

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

from stresser.data import load_word_stresses
from stresser.vectorization import SequenceVectorizer
from stresser.modeling import build_model

def eval_syll(gold, silver, syllab_only=False):
    # legend: 1 = begin stressed, 2 = begin unstressed, 3 = inside syllab
    results = []

    if not syllab_only:
        # unrelaxed evaluation
        for g, s in zip(gold, silver):
            for g_, s_ in zip(g, s):
                if g_ == s_:
                    results.append(1) 
                else:
                    results.append(0)
    else:
        # only look at token boundaries:
        for g, s in zip(gold, silver):
            for g_, s_ in zip(g, s):
                if g_ in (1, 2) or s_ in (1, 2):
                    if g_ in (1, 2) and s_ in (1, 2):
                        results.append(1) 
                    else:
                        results.append(0)
    
    return np.sum(results) / len(results)

def eval_word(gold, silver):
    return accuracy_score([str(s) for s in gold], 
                          [str(s) for s in silver])

def eval_levenshtein(gold, silver, norm=True):
    results = []
    for g, s in zip(gold, silver):
        g = ''.join([str(c) for c in g])
        s = ''.join([str(c) for c in s])
        if norm:
            results.append(Levenshtein.distance(g, s) / len(g))
        else:
            results.append(Levenshtein.distance(g, s))
        
    
    return np.mean(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='assets/stress_manually_reviewed.json')
    parser.add_argument('--train_size', default=0.9, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--rnd_state', default=23456, type=int)
    parser.add_argument('--min_cnt', default=0, type=int)
    parser.add_argument('--max_len', default=30, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--recurrent_dim', default=512, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--retrain', default=True, action='store_true')
    parser.add_argument('--model_prefix', default='word_analyzer', type=str)
    args = parser.parse_args()

    # is word frequency an issue?
    characters, labels = load_word_stresses(args.input_file)
    #characters = characters[:2000]
    #labels = labels[:2000]

    train_characters, rest_characters, train_labels, rest_labels = train_test_split(characters, labels,
                                                    train_size=args.train_size, random_state=args.rnd_state)
    dev_characters, test_characters, dev_labels, test_labels = train_test_split(rest_characters, rest_labels,
                                                 train_size=.5, random_state=args.rnd_state)

    print('num train:', len(train_characters))
    print('num dev:', len(dev_characters))
    print('num test:', len(test_characters))

    vectorizer = SequenceVectorizer(min_cnt=args.min_cnt, max_len=args.max_len)
    vectorizer.fit(train_characters)
    vectorizer.dump(args.model_prefix + '.vec')

    train_X = vectorizer.transform(train_characters)
    dev_X = vectorizer.transform(dev_characters)
    test_X = vectorizer.transform(test_characters)

    train_Y = vectorizer.normalize_label_len(train_labels)
    dev_Y = vectorizer.normalize_label_len(dev_labels)
    test_Y = vectorizer.normalize_label_len(test_labels)

    train_Y = to_categorical(train_Y, num_classes=4)
    dev_Y = to_categorical(dev_Y, num_classes=4)
    test_Y = to_categorical(test_Y, num_classes=4)

    print(train_X.shape)
    print(dev_X.shape)
    print(test_X.shape)

    print(train_Y.shape)
    print(dev_Y.shape)
    print(test_Y.shape)

    if args.retrain:
        model = build_model(vectorizer=vectorizer,
                            embed_dim=args.embed_dim,
                            num_layers=args.num_layers,
                            recurrent_dim=args.recurrent_dim,
                            lr=args.lr)
        model.summary()

        checkpoint = ModelCheckpoint(args.model_prefix + '.model', monitor='val_loss',
                                     verbose=1, save_best_only=True)
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                      patience=1, min_lr=0.00001,
                                      verbose=1, min_delta=0.001)

        try:
            model.fit(train_X, train_Y, validation_data=[dev_X, dev_Y],
                      epochs=args.num_epochs, batch_size=args.batch_size,
                      shuffle=True,
                      callbacks=[checkpoint, reduce_lr])
        except KeyboardInterrupt:
            pass

    # reload best model:
    vectorizer = SequenceVectorizer.load(args.model_prefix + '.vec')
    model = load_model(args.model_prefix + '.model')

    # calculate test accuracy
    test_loss, _= model.evaluate(test_X, test_Y)
    test_predictions = model.predict(test_X)

    print(test_Y.argmax(axis=-1)[0])
    print(test_predictions.argmax(axis=-1)[0])

    words, gold_decoded, silver_decoded = [], [], []

    for x, y, z in zip(test_X, test_Y, test_predictions):
        w = vectorizer.inverse_transform([x])[0]
        try:
            end = w.index('<PAD>')
        except ValueError:
            end = len(w)

        words.append(''.join(w[:end]))
        gold_decoded.append(list(y.argmax(axis=-1)[:end]))
        silver_decoded.append(list(z.argmax(axis=-1)[:end]))

    # reconvert to syllables and a stress pattern:
    syllables, stresses = [], []
    for w, g, s in zip(words, gold_decoded, silver_decoded):
        print(w)
        print(g)
        print(s)

        # syllabify
        sylls, stress, curr_syll = [], [], []
        for char, label in zip(w, s):
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
        print(sylls)
        print(stress)
        print('.................................................')

        syllables.append(sylls)
        stresses.append(stress)
    
    print('test loss:', test_loss)
    test_acc_token = eval_word(gold_decoded, silver_decoded)
    print('test acc (full token):', test_acc_token)

    test_acc_syll = eval_syll(gold_decoded, silver_decoded)
    print('test acc (syll-level):', test_acc_syll)

    test_acc_syll = eval_syll(gold_decoded, silver_decoded, syllab_only=True)
    print('test acc (syll, syllab only):', test_acc_syll)

    test_lev_acc = eval_levenshtein(gold_decoded, silver_decoded, norm=False)
    print('test acc (syll, levenshtein):', test_lev_acc)

    # wave list of errors for error analysis!

if __name__ == '__main__':
    main()

