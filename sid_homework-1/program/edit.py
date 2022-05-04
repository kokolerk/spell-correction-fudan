alphabet = set('abcdefghijklmnopqrstuvwxyz')

def edit1(word):
    '''
    编辑距离为1的候选
    :param word:
    :return:
    '''
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in alphabet]
    inserts = [L + c + R for L, R in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def edit2(word):
    '''
    编辑距离为2的候选，使用两次edit1即可
    :param word:
    :return:
    '''
    candidates=[]
    for w1 in edit1(word):
        for w2 in edit1(w1):
            candidates.append(w2)
    return set(candidates)

def edittype(word,error):
    '''
    判断两个单词之间的edit类别
    :param word:
    :param error:
    :return:
    '''
    word = word.lower()
    error = error.lower()
    for i in range(len(word)):
        if word == error[1:]:
            return error[0]+'|<s>', 'ins'
        if word[1:] == error:
            return '|'+word[0], 'del'
        if i >= len(error):
            return word[i]+'|'+error[i-1]+word[i], 'del'
        elif word[i] != error[i]:
            if word in [error[:i]+k+error[i:] for k in alphabet]:
                return error[i-1]+'|'+error[i-1]+word[i], 'del'
            elif word in [error[:i]+k+error[i+1:] for k in alphabet]:
                return error[i]+'|'+word[i], 'sub'
            elif word == error[:i]+error[i+1:] or word == error[:-1]:
                return word[i-1]+error[i]+'|'+word[i-1], 'ins'
            elif i+1 < len(word) and i+1 <len(error) and word[i]+ word[i+1] == error[i+1]+error[i] :
                return word[i+1]+word[i]+'|'+word[i]+word[i+1], 'trans'
    if len(word)<len(error):
        return word[-1]+error[-1]+'|'+word[-1], 'ins'
