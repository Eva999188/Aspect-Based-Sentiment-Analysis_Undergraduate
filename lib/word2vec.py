import pandas as pd
from gensim.models import Word2Vec
import jieba
import zhconv
import os
from utils import get_stop_word_set


# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 存储分词结果
def save_cut_word_rst(file_path):
    data = pd.read_csv(file_path + '.csv', usecols=['content'],lineterminator='\n')
    with open(file_path + '_cut_word_rst.txt', 'w', encoding='utf-8') as f_w:
        for content in data['content'].values:
            content = zhconv.convert(content.strip(), 'zh-cn')
            content = list(filter(lambda x: len(x.strip()) > 0, list(jieba.cut(content))))
            f_w.write(' '.join(content) + '\n')


for file_path in ['../data/theme',
                  '../data/story',
                  '../data/people']:
    save_cut_word_rst(file_path)

with open('../data/all_content_cut_word_rst.txt', 'w', encoding = 'utf-8') as f_w:
    for file_path in ['../data/theme',
                      '../data/story',
                      '../data/people']:
        with open(file_path + '_cut_word_rst.txt', encoding = 'utf-8') as f_r:
            for line in f_r:
                f_w.write(line)


class MySentences(object):
    def __init__(self, dirname, filter_ws):
        self.dirname = dirname
        self.filter_ws = filter_ws

    def __iter__(self):
        for line in open(self.dirname, encoding='utf-8'):
            yield list(filter(lambda x: x not in self.filter_ws, line.strip().split()))


# 词向量
sentences = MySentences('../data/all_content_cut_word_rst.txt', get_stop_word_set(only_punctuation=True))
model = Word2Vec(sentences, sg=1, size=100, compute_loss=True, window=5, workers=8, iter=8, min_count=0)
print(model.get_latest_training_loss(), len(model.wv.vocab))
model.save('../data/all_content_no_punc_100_8_mc2_fnl.w2v')


# 生成字向量
def save_char_content(save_path, fpath, stop_word_set):
    print('fpath:', fpath)
    data = pd.read_csv(fpath, usecols=['content'],lineterminator='\n')
    with open(save_path, 'a', encoding='utf-8') as f_w:
        for con in data['content'].values:
            f_w.write(' '.join(list(
                filter(lambda x: x not in stop_word_set and len(x.strip()) > 0, zhconv.convert(con, 'zh-cn')))) + '\n')


def save_char_content_single(fpath, stop_word_set):
    data = pd.read_csv(fpath, usecols=['content'],lineterminator='\n')
    fpath = fpath[:fpath.rfind('.')] + '_cut_char_rst.txt'
    print(fpath)
    with open(fpath, 'w', encoding='utf-8') as f_w:
        for con in data['content'].values:
            f_w.write(' '.join(list(
                filter(lambda x: x not in stop_word_set and len(x.strip()) > 0, zhconv.convert(con, 'zh-cn')))) + '\n')


all_csv_paths = ['../data/theme.csv',
                 '../data/story.csv',
                 '../data/people.csv']
stop_word_set = get_stop_word_set(only_punctuation=True)
save_char_path = '../data/all_char_content.txt'
for path in all_csv_paths:
    save_char_content(save_char_path, path, stop_word_set)
    save_char_content_single(path, stop_word_set)

sentences = MySentences(save_char_path, [])
model = Word2Vec(sentences, sg=1, size=100, compute_loss=True, window=10, workers=8, iter=15, min_count=0)
print(model.get_latest_training_loss())
print(len(model.wv.vocab))
model.save('../data/all_char_no_punc_100_15_mc2_fnl.w2v')
