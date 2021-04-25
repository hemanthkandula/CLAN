import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import collections
import argparse
import MeCab
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET

from vocab import Vocab

UNK_TOK = '<UNK>'
EOS_TOK = '<EOS>'
PAD_TOK = '<PAD>'


LANGS = ['en', 'fr', 'de', 'jp']
DOMS = ['books', 'dvd', 'music']
EXTRA_TOKENS = ['<EOS>', '<UNK>', '<PAD>']


ja_tagger = MeCab.Tagger("-Owakati")






def sample(x_list, n_samples, shuffle=True):
    if shuffle:
        size = x_list[0].size(0)
        perm = torch.randperm(size)
        return [x[perm][:n_samples] for x in x_list]
    else:
        return [x[:n_samples] for x in x_list]


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_vectors(path, maxload=-1):
    """

    """
    with open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split(' '))
        if maxload > 0:
            n = min(n, maxload)
        x = np.zeros([n, d])
        words = []
        for i, line in enumerate(fin):
            if i >= n:
                break
            tokens = line.rstrip().split(' ')
            words.append(tokens[0])
            x[i] = np.array(tokens[1:], dtype=float)

    return words, x


def shuffle(random_state, *args):
    """
    random_state: int
    args: List[Tensor]

    returns: List[Tensor]
    """
    torch.manual_seed(random_state)
    size = args[0].size(0)
    perm = torch.randperm(size)
    res = [x[perm] for x in args]
    return res


def load_senti_corpus(path, vocab, encoding='utf-8', maxlen=512, random_state=None, labels=['__pos__', '__neg__']):
    """
    path: str
    vocab: Vocab
    encoding: str
    maxlen: int
    random_state: int
    labels: List[str]

    returns: LongTensor of shape (size, maxlen), LongTensor of shape (size,)
    """
    corpus, y = [], []
    l2i = {l: i for i, l in enumerate(labels)}
    with open(path, 'r', encoding=encoding) as fin:
        for line in fin:
            label, text = line.rstrip().split(' ', 1)
            y.append(l2i[label])
            corpus.append([vocab.w2idx[w] if w in vocab else vocab.w2idx[UNK_TOK]
                           for w in text.split(' ')] + [vocab.w2idx[EOS_TOK]])
    size = len(corpus)
    X = torch.full((size, maxlen), vocab.w2idx[PAD_TOK], dtype=torch.int64)
    l = torch.empty(size, dtype=torch.int64)
    y = torch.tensor(y)
    for i, xs in enumerate(corpus):
        sl = min(len(xs), maxlen)
        l[i] = sl
        X[i, :sl] = torch.tensor(xs[:sl])
    if random_state is not None:
        X, y, l = shuffle(random_state, X, y, l)
    return X, y, l


def load_lm_corpus(path, vocab, encoding='utf-8', random_state=None):
    """
    path: str
    vocab: Vocab
    encoding: str
    random_state: int (optional)

    returns: torch.LongTensor of shape (corpus_size,)
    """
    if random_state is None:
        # first pass: count the number of tokens
        with open(path, 'r', encoding=encoding) as f:
            ntokens = 0
            for line in f:
                words = line.rstrip().split(' ') + [EOS_TOK]
                ntokens += len(words)

        # second pass: convert tokens to ids
        with open(path, 'r', encoding=encoding) as f:
            ids = torch.LongTensor(ntokens)
            p = 0
            for line in f:
                words = line.rstrip().split(' ') + [EOS_TOK]
                for w in words:
                    if w in vocab.w2idx:
                        ids[p] = vocab.w2idx[w]
                    else:
                        ids[p] = vocab.w2idx[UNK_TOK]
                    p += 1

    else:
        with open(path, 'r', encoding=encoding) as f:
            corpus = [line.rstrip() + ' ' + EOS_TOK for line in f]

        ntokens = sum(len(line.split(' ')) for line in corpus)
        ids = torch.LongTensor(ntokens)
        p = 0
        np.random.seed(random_state)
        for i in np.random.permutation(len(corpus)):
            for w in corpus[i].split(' '):
                if w in vocab.w2idx:
                    ids[p] = vocab.w2idx[w]
                else:
                    ids[p] = vocab.w2idx[UNK_TOK]
                p += 1

    return ids


def load_vectors_with_vocab(path, vocab, maxload=-1):
    """
    path: str
    vocab: Vocab
    maxload: int
    """
    count = 0
    with open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split(' '))
        x = np.zeros([len(vocab), d])
        words = []
        for i, line in enumerate(fin):
            if maxload > 0 and i >= maxload:
                break
            tokens = line.rstrip().split(' ')
            if tokens[0] in vocab:
                x[vocab.w2idx[tokens[0]]] = np.array(tokens[1:], dtype=float)
                count += 1
    return x, count


def load_lexicon(path, src_vocab, trg_vocab, encoding='utf-8', verbose=False):
    """
    path: str
    src_vocab: Vocab
    trg_vocab: Vocab
    encoding: str
    verbose: bool

    returns: collections.defautldict
    """
    lexicon = collections.defaultdict(set)
    vocab = set()
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            src, trg = line.rstrip().split()
            if src in src_vocab and trg in trg_vocab:
                lexicon[src_vocab.w2idx[src]].add(trg_vocab.w2idx[trg])
            vocab.add(src)
    if verbose:
        print('[{}] OOV rate = {:.4f}'.format(path, 1 - len(lexicon) / len(vocab)))

    return lexicon, len(vocab)



def get_batch(source, i, bptt, seq_len=None, evaluation=False, batch_first=False, cuda=False):
    assert isinstance(bptt, int)

    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    if batch_first:
        data = source[i:i + seq_len].t()
        target = source[i + 1:i + 1 + seq_len].t().contiguous().view(-1)
    else:
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)

    if cuda:
        data = data.cuda()
        target = target.cuda()

    return data, target

def tokenize(sent, lang):
    if lang == 'ja':
        toks = ja_tagger.parse(sent).split()
    else:
        toks = word_tokenize(sent)
    toks = [t.lower() for t in toks]
    toks = ['<NUM>' if t.isdigit() else t for t in toks]
    return toks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='../../data/cls-acl10-unprocessed/', help='input directory')
    parser.add_argument('--output_dir', default='../../data/', help='output directory')
    parser.add_argument('--vocab_cutoff', type=int, default=15000, help='maximum vocab size')
    parser.add_argument('--maxlen', type=int, default=256, help='maximum length for each labeled example')
    parser.add_argument('--val_size', type=int, default=600, help='size of the validation set')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    print('Configuration:')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    print()

    output_dir = args.output_dir
    tokenized_dir = os.path.join(output_dir, 'tokenized')
    vocab_dir = os.path.join(output_dir, 'vocab')

    for d in [output_dir, tokenized_dir, vocab_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    print()
    print('Tokenizing raw text...')
    for lang in LANGS:
        f_lang = os.path.join(tokenized_dir, f'{lang}.unlabeled')
        f_lang = open(f_lang, 'w', encoding='utf-8')

        for dom in DOMS:
            f_lang_dom = os.path.join(tokenized_dir, f'{lang}.{dom}.unlabeled')
            f_lang_dom = open(f_lang_dom, 'w', encoding='utf-8')

            for part in ['train', 'test', 'unlabeled']:
                if part != 'unlabeled':
                    f_label = os.path.join(tokenized_dir, f'{lang}.{dom}.{part}')
                    f_label = open(f_label, 'w', encoding='utf-8')

                fname = os.path.join(args.input_dir, 'jp' if lang == 'ja' else lang, dom, f'{part}.review')
                root = ET.parse(fname).getroot()
                nitem, npos, nneg = 0, 0, 0
                for t in root:
                    try:
                        dic = {x.tag: x.text for x in t}
                        tokens = tokenize(str(dic['text']), lang)
                        if part != 'unlabeled':
                            if float(dic['rating']) > 3:
                                label = '__pos__'
                                npos += 1
                            else:
                                label = '__neg__'
                                nneg += 1
                            f_label.write(label + ' ' + ' '.join(tokens) + '\n')
                        if part != 'test':
                            f_lang.write(' '.join(tokens) + '\n')
                            f_lang_dom.write(' '.join(tokens) + '\n')
                        nitem += 1
                    except Exception as e:
                        print('[ERROR] ignoring item - {}'.format(e))

                if part != 'unlabeled':
                    f_label.close()

                print('file: {:60}\tvalid: {:7}\tpos: {:5}\tneg: {:5}'.format(fname, nitem, npos, nneg))

            f_lang_dom.close()
        f_lang.close()

    print()
    print('Generating vocabularies...')
    for lang in LANGS:
        with open(os.path.join(tokenized_dir, f'{lang}.unlabeled'), 'r', encoding='utf-8') as fin:
            corpus = [row.rstrip() for row in fin]
        vocab = Vocab(corpus)
        vocab.cutoff(args.vocab_cutoff - len(EXTRA_TOKENS))
        for tok in EXTRA_TOKENS:
            vocab.add_word(tok)
        vocab.save(os.path.join(vocab_dir, f'{lang}.vocab'))
        print('{} vocab of size {} generated'.format(lang, len(vocab)))

    print()
    print('Binarizing data...')
    unlabeled_set = {lang: {} for lang in LANGS}
    train_set = {lang: {} for lang in LANGS}
    val_set = {lang: {} for lang in LANGS}
    test_set = {lang: {} for lang in LANGS}
    for lang in LANGS:
        # load vocab
        vocab = Vocab(path=os.path.join(vocab_dir, f'{lang}.vocab'))
        unlabeled_set[lang]['vocab'] = train_set[lang]['vocab'] = val_set[lang]['vocab'] = test_set[lang]['vocab'] = vocab

        # load unlabeled data from a language
        x = load_lm_corpus(os.path.join(tokenized_dir, f'{lang}.unlabeled'), vocab, random_state=args.seed)
        unlabeled_set[lang]['unlabeled'] = x
        size = x.size(0)
        print('[{}]\tsize = {}'.format(lang,  size))
        print('[{}]\tOOV rate = {:.2f}'.format(lang, x.bincount()[vocab.w2idx[UNK_TOK]].item() / size))

        for dom in DOMS:
            # load unlabeled data from a language-domain pair
            x = load_lm_corpus(os.path.join(tokenized_dir, f'{lang}.{dom}.unlabeled'), vocab, random_state=args.seed)
            size = x.size(0)
            unlabeled_set[lang][dom] = x
            print('[{}_{}]\tunlabeled size = {}'.format(lang, dom,  size))
            print('[{}_{}]\tOOV rate = {:.2f}'.format(lang, dom, x.bincount()[vocab.w2idx[UNK_TOK]].item() / size))

            # load train / test data
            train_x, train_y, train_l = load_senti_corpus(os.path.join(tokenized_dir,  f'{lang}.{dom}.train'),
                                                          vocab, maxlen=args.maxlen, random_state=args.seed)
            test_x, test_y, test_l = load_senti_corpus(os.path.join(tokenized_dir,  f'{lang}.{dom}.test'),
                                                       vocab, maxlen=args.maxlen, random_state=args.seed)
            train_set[lang][dom] = [train_x, train_y, train_l]
            val_set[lang][dom] = [val_x, val_y, val_l] = sample([train_x, train_y, train_l], args.val_size)
            test_set[lang][dom] = [test_x, test_y, test_l]
            print('[{}_{}]\ttrain size = {}'.format(lang, dom,  train_x.size(0)))
            print('[{}_{}]\tval size = {}'.format(lang, dom,  val_x.size(0)))
            print('[{}_{}]\ttest size = {}'.format(lang, dom, test_x.size(0)))

    f_unlabeled = os.path.join(output_dir, 'unlabeled.pth.tar')
    f_train = os.path.join(output_dir, 'train.pth.tar')
    f_val = os.path.join(output_dir, 'val.pth.tar')
    f_test = os.path.join(output_dir, 'test.pth.tar')
    torch.save(unlabeled_set, f_unlabeled)
    torch.save(val_set, f_val)
    torch.save(train_set, f_train)
    torch.save(test_set, f_test)
    print('unlabeled data saved to {}'.format(f_unlabeled))
    print('training data saved to {}'.format(f_train))
    print('validation data saved to {}'.format(f_val))
    print('test data saved to {}'.format(f_test))


if __name__ == '__main__':
    main()

