import re
import nltk
from dataloader import *
from edit import *
from eval import evaluation


editerror,testdata =load_testdata() #加载测试数据
vocab=load_vocab() #加载词典
edit_dict=load_edit() #编辑距离概率

# #加载ans.txt作为训练集
# unigram_dict_ans=load_ngram(1) #一元文法词典+概率
# bigram_dict_ans=load_ngram(2) #二元文法词典+概率
# trigram_dict_ans=load_ngram(3) #三元文法词典+概率
#
# #加载外部语料作为训练
# unigram_dict=load_unigram() #一元文法词典+概率
# bigram_dict=load_bigram() #二元文法词典+概率

class spell_correction():

    def __init__(self,LM,channel,editdistance,corpus):
        self.LM=LM
        self.channel=channel
        self.editdistance=editdistance
        self.corpus=corpus

        # 定义LM词典
        if corpus==True:
            self.ngram_dict=load_ngram(int(LM))
        elif LM=='1':
            self.ngram_dict=load_unigram()
        else:
            self.ngram_dict=load_bigram()

    #查找句子里面的词汇是否在词典里
    def find_in_vocab(self,words):
        return set(word for word in words if word in vocab)

    #判断词汇的错误类型
    def edit_type(self, candidate, word):
        wrong_type = edittype(candidate,word) # 和上面的不同
        if wrong_type == None: # w|w的情况
            return np.log(0.95)
        elif wrong_type[0] in edit_dict.keys():
            return edit_dict[wrong_type[0]]
        else:
            return -100000

    #返回LM-prob
    def ngram_prob(self,s):
        if s in self.ngram_dict.keys():
            return self.ngram_dict[s]
        else:
            return -100000

    #计算候选单词在句子里面的LM-prob
    def sentence_prob(self,candidate,j,sentence):
        if self.LM =='1':
            return self.ngram_prob(candidate)
        elif self.LM =='2':
            if j == 0:
                return self.ngram_prob(candidate + ' ' + sentence[j + 1].lower())
            else:
                return self.ngram_prob(candidate + ' ' + sentence[j + 1].lower()) + self.ngram_prob(sentence[j - 1].lower() + ' ' + candidate)
        elif self.LM=='3':
            if j == 0 and j + 2 < len(sentence):
                return self.ngram_prob(candidate + ' ' + sentence[j + 1].lower() + ' ' + sentence[j + 2].lower())
            elif j == len(sentence) - 1 and j - 2 > -1:
                return self.ngram_prob(sentence[j - 2] + ' ' + sentence[j - 1].lower() + ' ' + candidate)
            elif j == 1 and j + 2 < len(sentence):
                return self.ngram_prob(candidate + ' ' + sentence[j + 1].lower() + ' ' + sentence[j + 2].lower()) + \
                       self.ngram_prob(sentence[j - 1].lower() + ' ' + candidate + ' ' + sentence[j + 1].lower())
            elif j == len(sentence) - 2 and j - 2 > -1:
                return self.ngram_prob(sentence[j - 2] + ' ' + sentence[j - 1].lower() + ' ' + candidate) + \
                       self.ngram_prob(sentence[j - 1] + ' ' + candidate + ' ' + sentence[j + 1])
            else:
                if len(sentence) == 3:
                    return self.ngram_prob(sentence[j - 1] + ' ' + candidate + ' ' + sentence[j + 1])
                else:
                    return self.ngram_prob(sentence[j - 2] + ' ' + sentence[j - 1].lower() + ' ' + candidate) + \
                           self.ngram_prob(sentence[j - 1] + ' ' + candidate + ' ' + sentence[j + 1]) + \
                           self.ngram_prob(candidate + ' ' + sentence[j + 1] + ' ' + sentence[j + 2])
        else:
            print(' wrong LM type and corpus type!')
            return 0

    # 非词纠正
    def non_word_correct(self,sentence):
        '''
        非词拼写错误纠正
        :param sentence: 还有错误的句子
        :return: wrong num 检测处理的错误数
        '''
        wrong = 0
        for j in range(len(sentence)):
            word = sentence[j]
            if bool(re.search(r"[\d.,/'-]", word)) or word.lower() in vocab:
                continue
            word_lower = word.lower()
            candidates = self.find_in_vocab(edit1(word_lower))
            if self.editdistance==2 and len(candidates)==0: candidates = self.find_in_vocab(edit2(word_lower))
            p_flag = -2e5
            right = word
            for candidate in candidates:
                if self.channel==False:
                    p =self.sentence_prob(candidate, j, sentence)
                else :
                    p = self.sentence_prob(candidate, j, sentence) + self.edit_type(candidate, word_lower) #channel model for edits

                if p > p_flag:
                    p_flag = p
                    right = candidate
            # 还原大小写
            if not word.islower():
                flag = 0
                for each in word:
                    flag += int(each.isupper())
                if flag == 1:
                    right = right[0].upper() + right[1:]
                else:
                    right = right.upper()
            sentence[j] = right  # to do supper letters
            wrong += 1
        return wrong

    #词纠正
    def real_word_correct(self,sentence):
        '''
        词拼写错误纠正
        :param sentence:
        :return:
        '''
        for j in range(len(sentence)):
            word = sentence[j]
            if bool(re.search(r"[\d.,/'-]", word)):
                continue
            # edit distance = 1
            word_lower = word.lower()
            candidates = self.find_in_vocab(edit1(word_lower))
            if self.editdistance==2 and len(candidates) == 0: candidates = spell_correction.find_in_vocab(edit2(word_lower))
            # 加入原本的单词
            candidates.add(word_lower)
            p_flag = -2e5
            right = word_lower
            for candidate in candidates:
                if self.channel == False:
                    p = self.sentence_prob(candidate, j, sentence)
                elif self.channel == True:
                    p = self.sentence_prob(candidate, j, sentence) +self.edit_type(candidate, word_lower)  # channel model for edits1
                else:
                    print("wrong LM and Channel model choices ! ")
                    return
                if p > p_flag:
                    p_flag = p
                    right = candidate
            #         print(candidates,right)
            if right == word_lower:
                continue
            # 还原大小写
            if word != word_lower:
                flag = 0
                for each in word:
                    flag += int(each.isupper())
                if flag == 1:
                    right = right[0].upper() + right[1:]
                else:
                    right = right.upper()
            sentence[j] = right
            if right.lower() != word_lower:
                return None

    def word_correct(self):
        '''
        语法拼写纠正
        :return:
        '''
        sentences=[]
        for i in range(1000):
            sentence = testdata[i]
            wrong = self.non_word_correct(sentence) #先进行non-word拼写纠正
            if wrong < editerror[i]: #进行real-word拼写纠正
                self.real_word_correct(sentence)
            sentences.append(sentence)
        file = open('./result.txt', 'w')
        for index in range(1000):
            file.write(str(index + 1) + '\t' + ' '.join(sentences[index]) + '\n')
        file.close()



