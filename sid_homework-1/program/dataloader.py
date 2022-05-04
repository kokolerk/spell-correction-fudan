import nltk
from collections import defaultdict
import numpy as np
import re
testdata_path='./testdata.txt'
vocab_path='./vocab.txt'
singleEdit_path='./single-edit.txt'
unigram_path='./count-1w.txt'
bigram_path='./count-2w.txt'
corpus_file_path='./ans.txt'

#加载待修改的错误文本
def load_testdata(testdata_path=testdata_path):
    '''
    读取数据testdata的数据
    使用nltk tokenization
    :return:
    '''
    file=open(testdata_path,'r')
    editerror=[]
    data=[]
    for i in range(1000):
        line=file.readline().split('\t')
        editerror.append(int(line[1]))
        sentence= nltk.word_tokenize(line[2])
        data.append(sentence)
    return editerror,data

#加载edit的统计数据
def load_edit(singleEdit_path=singleEdit_path):
    '''
    读取单个字符的spell错误数据，log一下数据
    :return:
    '''
    f= open(singleEdit_path, 'r')
    edit_dict= {}
    l = f.readlines()
    for each in l:
        each = each.strip().split('\t')
        edit_dict[each[0]] = np.log(int(each[1]))
    return edit_dict

#加载vocab.txt，词汇表
def load_vocab(vocab_path=vocab_path):
    '''
    读取词典
    去除数字、特殊符号
    大写转化为小写
    :return:
    '''
    vocab = []
    with open(vocab_path,'r') as file :
        for line in file.readlines():
            word=re.sub('[0-9,.!“”‘’/\—\–\-\'\"]', '', line.strip())
            if word !='':
                vocab.append(word)
    return set([each.lower() for each in vocab if each!=''])

# 加载ans.txt作为训练集
def load_ngram(n,corpus_file_path=corpus_file_path):
    f= open(corpus_file_path,'r')
    overall_low=[]
    for i in range(1000):
        tokens=f.readline().split('\t')[1]
        tokens=nltk.word_tokenize(tokens)
        tokens_low=[]
        for word in tokens :
            word_low=re.sub('[,.!“”‘’\—\–\'\"]', '', word).lower()
            if word_low!='':
                tokens_low.append(word_low)
        overall_low.append(tokens_low)
    output = defaultdict()
    for l in overall_low:
        for i in range(len(l)-n+1):
            temp = " ".join(l[i:i+n])
            if temp not in output:
                output[temp] = 0
            output[temp] += 1
    SUM = sum(output.values())
    for each in output:
        output[each] = np.log(output[each] / SUM)
    return output

#加载count-1w作为训练集
def load_unigram(unigram_path=unigram_path):
    '''

    :return:
    '''
    ret=defaultdict()
    with open(unigram_path,'r') as file:
        for line in file.readlines():
            line=line.strip().split('\t')
            word_low = re.sub('[0-9,.!“”‘’\—\–\'\"]', '', line[0]).lower()
            if word_low!='':
                ret[word_low]=np.log(int(line[1]))
    return ret

#加载count-2w作为训练集
def load_bigram(bigram_path=bigram_path):
    '''
    读取bigram词典
    :return:
    '''
    ret = defaultdict()
    with open(bigram_path, 'r') as file:
        for line in file.readlines():
            line = line.strip().split('\t')
            words=line[0].split(' ')
            pre=re.sub('[,.!“”‘’\—\–\'\"]', '', words[0]).lower()
            pre_digit=re.sub('[0-9]','',pre).lower()
            if pre_digit!=pre:
                continue
            now=re.sub('[,.!“”‘’\—\–\'\"]', '', words[1]).lower()
            now_digit = re.sub('[0-9]', '', now).lower()
            if now_digit != now:
                continue
            if pre !='' and now!='':
                ret[pre + ' ' + now] = np.log(int(line[1]))
    return ret