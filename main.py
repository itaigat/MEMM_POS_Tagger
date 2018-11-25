from src.utils.decoder import Comp
from src.utils.common import create_all_features
from src.utils.params import Params
import os


bla = Comp("C:\\Users\\Itai Gat\\Documents\\Technion\\19A - LAST\\NLP\MEMM_POS_Tagger\\resources\\train.wtag")
for tuples, sentence in bla:
    create_all_features(Params.feature_functions, tuples, sentence)
