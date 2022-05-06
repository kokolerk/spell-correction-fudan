import nltk
from nltk.corpus import reuters
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
        edit_dict[each[0]] = int(each[1])
    SUM = sum(edit_dict.values())
    for each in edit_dict.keys():
        edit_dict[each] = np.log(edit_dict[each] / SUM)
    # path='./edit_prob.txt'
    # file=open(path,'w')
    # for each in edit_dict.keys():
    #     file.write(each+'\t'+str(edit_dict[each])+'\n')

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
    return output  # 大概范围是-8。-9。-5之类的

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
    # SUM = sum(ret.values())
    # for each in ret:
    #     ret[each] = np.log(ret[each] / SUM)
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
            if pre=='<s>': #开头去掉
                continue
            pre_digit=re.sub('[0-9]','',pre).lower()
            for i in pre: #特殊的奇怪符号去掉
                if i<'a' or i>'z':
                    continue
            if pre_digit!=pre:
                continue
            now=re.sub('[,.!“”‘’\—\–\'\"]', '', words[1]).lower()
            now_digit = re.sub('[0-9]', '', now).lower()
            if now_digit != now:
                continue
            if pre !='' and now!='':
                ret[pre + ' ' + now] = np.log(int(line[1]))
    SUM = sum(ret.values())
    for each in ret:
        ret[each] = np.log(ret[each] / SUM)*0.5
    return ret

#加载geuter作为训练集
def bigram_reuters():
    # 读取语料库
    categories = reuters.categories()  # 路透社语料库的类别
    # print(categories)
    corpus = reuters.sents(categories=categories)  # sents()指定分类中的句子
    # print(corpus)

    # 构建语言模型：bigram
    term_count = {}
    bigram_count = {}
    for docs in corpus:
        docs = ['<s>'] + docs
        doc = []
        for word in docs:
            word_low = re.sub('[,.!“”‘’\—\–\'\"]', '', word).lower()
            # if bool(re.search(r"[\d.,/'-]", word_low)) or len(word_low)==1:
            #     continue
            if word_low != '':
                doc.append(word_low)
        # '<s>'表示开头
        for i in range(0, len(doc) - 1):
            bigram = doc[i:i + 2]  # bigram为第i,i+1个单词组成的
            bigram = ' '.join(bigram)
            if bigram in bigram_count:
                bigram_count[bigram] += 1
            else:
                bigram_count[bigram] = 1
    SUM = sum(bigram_count.values())
    for each in bigram_count.keys():
        bigram_count[each] = np.log( bigram_count[each] / SUM)
    # path='./bigram_reuters.txt'
    # file=open(path,'w')
    # for each in bigram_count.keys():
    #     file.write(each+'\t'+str(bigram_count[each])+'\n')
    return bigram_count

#加载geuter作为训练集
def unigram_reuters():
    # 读取语料库
    categories = reuters.categories()  # 路透社语料库的类别
    corpus = reuters.sents(categories=categories)  # sents()指定分类中的句子
    # 构建语言模型：unigram
    term_count = {}
    for docs in corpus:
        doc=[]
        for word in docs:
            word_low = re.sub('[,.!“”‘’\—\–\'\"]', '', word).lower()
            if bool(re.search(r"[\d.,/'-]", word_low)) or len(word_low)==1:
                continue
            if word_low != '':
                doc.append(word_low)
        for i in range(0, len(doc)):
            term = doc[i]  # term是doc中第i个单词
            if term in term_count:
                term_count[term] += 1  # 如果term存在term_count中，则加1
            else:
                term_count[term] = 1  # 如果不存在，则添加，置为1
    SUM = sum(term_count.values())
    for each in term_count.keys():
        term_count[each] = np.log(term_count[each] / SUM)

    # path='./unigram_reuters.txt'
    # file=open(path,'w')
    # for each in term_count.keys():
    #     file.write(each+'\t'+str(term_count[each])+'\n')

    return term_count

#加载geuter作为训练集
def trigram_reuters():
    # 读取语料库
    categories = reuters.categories()  # 路透社语料库的类别
    corpus = reuters.sents(categories=categories)  # sents()指定分类中的句子
    # 构建语言模型：unigram
    term_count = {}
    for docs in corpus:
        doc=[]
        for word in docs:
            word_low = re.sub('[,.!“”‘’\—\–\'\"]', '', word).lower()
            if bool(re.search(r"[\d.,/'-]", word_low)) or len(word_low)==1:
                continue
            if word_low != '':
                doc.append(word_low)
        for i in range(0, len(doc)-2):
            term = doc[i:i+3]  # term是doc中第i个单词
            term = ' '.join(term)
            if term in term_count:
                term_count[term] += 1  # 如果term存在term_count中，则加1
            else:
                term_count[term] = 1  # 如果不存在，则添加，置为1
    SUM = sum(term_count.values())
    for each in term_count.keys():
        term_count[each] = np.log(term_count[each] / SUM)

    path='./trigram_reuters.txt'
    file=open(path,'w')
    for each in term_count.keys():
        file.write(each+'\t'+str(term_count[each])+'\n')

    return term_count

trigram_reuters()
