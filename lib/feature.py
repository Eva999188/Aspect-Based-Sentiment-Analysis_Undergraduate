import numpy as np
import pickle

import xlwt


def get_data(file_path):
    data = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())
    return data


train_content_ori = get_data(
    '../data/theme_cut_word_rst.txt')
val_content_ori = get_data(
    '../data/story_cut_word_rst.txt')
test_content_ori = get_data(
    '../data/people_cut_word_rst.txt')

print(len(train_content_ori), len(val_content_ori), len(test_content_ori))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectorizer.fit(train_content_ori)

train_content = vectorizer.transform(train_content_ori)
val_content = vectorizer.transform(val_content_ori)
test_content = vectorizer.transform(test_content_ori)

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=20 * 4, n_iter=7, random_state=2018)
svd.fit(train_content)

train_svd = svd.transform(train_content)
val_svd = svd.transform(val_content)
test_svd = svd.transform(test_content)

workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('My Worksheet')
workbook.save(r'../data/test0526.xls')
train_svd = list(train_svd)
cnt1 = 0
cnt2 = 0
for v in train_svd:
   cnt2 = 0
   for i in v:
       worksheet.write(cnt1, cnt2, label=str(i))
       cnt2 = cnt2 + 1
   cnt1 = cnt1 + 1
workbook.save(r'../data/test0526.xls')

prefix = 'svd_tfidf_withP_80'
np.save('../data/%s_train' % prefix, train_svd)
np.save('../data/%s_val' % prefix, val_svd)
np.save('../data/%s_test' % prefix, test_svd)
pickle.dump(svd, open('../data/%s.pk' % prefix, 'wb'))
