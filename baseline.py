import argparse
import os
import subprocess
import shutil

import stresser.utils as u
import pycrfsuite

def vectorize(words_, labels_, window):
    X, Y = [], []

    for word, labels in zip(words_, labels_):
        seq_X, seq_Y = [], []

        word = ['<BOS>'] + word + ['<EOS>']
        labels = ['<BOS>'] + labels + ['<EOS>']

        for idx, (char, lab) in enumerate(zip(word, labels)):
            # skip BOS/EOS
            if idx in (0, len(word) - 1):
                continue
            
            left_context = [word[idx-(t+1)] for t in range(window) if idx-(t+1) >= 0][::-1]
            while len(left_context) < window:
                    left_context = ['<PAD>'] + left_context

            right_context = [word[idx+t+1] for t in range(window) if idx+t+1 < len(word)]
            while len(right_context) < window:
                    right_context += ['<PAD>']
            
            features = left_context + [char] + right_context
            features = [str(i + 1)+'-' + f for i, f in enumerate(features)]
            
            seq_X.append(features)
            seq_Y.append(str(labels[idx]))
        
        X.append(seq_X)
        Y.append(seq_Y)
    
    return X, Y


def main():
    parser = argparse.ArgumentParser(description='Trains baseline model')
    parser.add_argument('--input_dir', type=str,
                        default='data/splits',
                        help='location of the splits folder')
    parser.add_argument('--baseline_dir', type=str,
                        default='model_b',
                        help='location of the model folder')
    parser.add_argument('--window', type=int,
                        default=3,
                        help='Length of window around focus character (symmetric)')
    parser.add_argument('--retrain', default=False, action='store_true',
                        help='Retrain a model from scratch')
    args = parser.parse_args()
    print(args)

    train, dev, test = u.load_splits(args.input_dir)

    train_words, train_Y = train
    dev_words, dev_Y = dev
    test_words, test_Y = test

    train_X, train_Y = vectorize(train_words, train_Y, window=args.window)
    dev_X, dev_Y = vectorize(dev_words, dev_Y, window=args.window)
    test_X, test_Y = vectorize(test_words, test_Y, window=args.window)

    m_path = f'{args.baseline_dir}/model.crfsuite'

    if args.retrain:
        try:
            shutil.rmtree(args.baseline_dir)
        except FileNotFoundError:
            pass
        os.mkdir(args.baseline_dir)
    
        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(train_X, train_Y):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': 1.0, 
            'c2': 1e-3,
            'max_iterations': 60,
            'feature.possible_transitions': True
        })

        trainer.train(m_path)

    tagger = pycrfsuite.Tagger()
    tagger.open(m_path)

    dev_silver = [tagger.tag(x) for x in dev_X]
    with open(os.sep.join((args.baseline_dir, 'silver_dev.json')), 'w') as f:
        f.write(u.jsonify(dev_words, dev_silver, rm_symbols=False))

    test_silver = [tagger.tag(x) for x in test_X]
    with open(os.sep.join((args.baseline_dir, 'silver_test.json')), 'w') as f:
        f.write(u.jsonify(test_words, test_silver, rm_symbols=False))


if __name__ == '__main__':
    main()