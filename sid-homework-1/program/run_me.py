from spell_correct2 import spell_correction
from eval import evaluation

if __name__ == "__main__":
    # best config
    s=spell_correction(LM='3', corpus=False,channel=True,editdistance=1,defauttpro=-1e3,lamda=2)
    s.word_correct()
    s.real_word_correct('')
    evaluation()
