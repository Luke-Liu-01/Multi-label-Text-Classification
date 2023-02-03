import re
import jieba
import torch
from torchtext.legacy import data
from torchtext.vocab import Vectors

BATCH_SIZE = 32
DEVICE = 0 if torch.cuda.is_available() else -1


# Jieba tokenizer
def Tokenizer(text):
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    text = regex.sub(' ', text)  # filter all the non-Chinese words (replace them with space)
    return [word for word in jieba.cut(text) if word.strip()]


# convert labels into one-hot vectors
def OneHotLabel(label):
    label = list(map(int, label.split('-')))  # string -> list
    label = torch.Tensor(label)  # list -> Tensor
    return label


# load stopwords
def GetStopWords():
    stop_words = []
    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stop_words.append(line.strip())
    return stop_words


class MultiLabelDataSet():

    def __init__(self, mode):
        self.mode = mode  # static: using pre-trained word2vec model; rand: random initialization
        stop_words = GetStopWords()

        # text pre-processing configurations
        self.text = data.Field(sequential=True, tokenize=Tokenizer, stop_words=stop_words)
        self.label = data.Field(sequential=True, tokenize=OneHotLabel, use_vocab=False)

        self.train_set, self.valid_set = data.TabularDataset.splits(
            path='data/',
            skip_header=True,
            train='multi_label_train.csv',
            validation='multi_label_valid.csv',
            format='csv',
            fields=[('label', self.label), ('text', self.text)]
        )

        # using pre-trained word2vec model
        # non-static: fine-tune during training
        if self.mode == 'static' or self.mode == 'not-static':
            cache = 'data/.vector_cache'
            vectors = Vectors(name='data/sgns.zhihu.word', cache=cache)  # Zhihu_QA
            self.text.build_vocab(self.train_set, self.valid_set, vectors=vectors)
            self.embedding_dim = self.text.vocab.vectors.size()[-1]
            self.vectors = self.text.vocab.vectors
        else:
            self.text.build_vocab(self.train_set, self.valid_set)
            self.embedding_dim = 300
            self.vectors = None

        self.vocab_num = len(self.text.vocab)  # vocabulary size (the number of different words)
        self.label_num = 18  # 18 labels


    def GetIter(self):
        train_iter, val_iter = data.Iterator.splits(
            (self.train_set, self.valid_set),
            sort_key=lambda x: len(x.text),
            # sort=False,
            batch_sizes=(BATCH_SIZE, len(self.valid_set)),
            device=DEVICE
        )
        return train_iter, val_iter

    def GetArgs(self):
        args = (self.mode, self.vocab_num, self.label_num, self.embedding_dim, self.vectors)
        return args