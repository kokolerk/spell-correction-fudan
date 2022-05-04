from spell_correct import spell_correction
from eval import evaluation

if __name__ == "__main__":
    # best config
    s=spell_correction(LM='2', corpus=True,channel=False,editdistance=2)
    s.word_correct()
    evaluation()