from sentence_transformers import SentenceTransformer
import numpy as np
import re

pre_trained_model = 'bert-base-nli-mean-tokens'

class SbertEncoder:
    
    def __init__(self):
        self.sbert = SentenceTransformer(pre_trained_model)
        
    def encode(self, x):
        pass
    
    
class SbertSentenceMeanEncoder(SbertEncoder):
    
    def __init__(self):
        SbertEncoder.__init__(self)
    
    def encode(self, x):
        sentences = re.split('!+ |\?+ |\.+ |!|\?|\.', x)
        sentences = list(filter(None, sentences))
        if len(sentences) == 0:
            mean_encoding = self.sbert.encode(x)
        else: 
            encosings = self.sbert.encode(sentences)
            mean_encoding = np.array(encosings).mean(axis=0)
        return mean_encoding

class SbertFullTextEncoder(SbertEncoder):
    
    def __init__(self):
        SbertEncoder.__init__(self)
        self.sbert.max_seq_length = 512
    
    def encode(self, x):
        encosing = self.sbert.encode(x)
        return encosing