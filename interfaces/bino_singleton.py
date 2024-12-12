from binoculars import Binoculars
import torch

'''
Module are only evaluated once, thus it will be the same instance of Binoculars in every module
that will call this one
'''

BINO = Binoculars()
TOKENIZER = BINO.tokenizer