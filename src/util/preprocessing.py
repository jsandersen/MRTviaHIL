import re
from tqdm import tqdm

def preprocess_corpus(X, keep_nbr=False):
    return [preprocess(x, keep_nbr) for x in tqdm(X)]

def preprocess(text, keep_nbr=False):
    
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'<.*?>', ' ', text)
    
    text = re.sub(r'(?<=\d)[,\.]', '', text)
    text = re.sub(r'(\.\d+)', '', text)
    text = re.sub(r"\b\.*\d+\b", "", text)
    text = re.sub(r"\b\.+\d+\b", "", text)
    
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\w+(\.\w+)+.',' ', text)
    
    text = re.sub(r'&#34', '', text)
    
    text = text.strip().lower()
    
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)

    text = re.sub(r"i'm",   "i am", text)
    text = re.sub(r"he's",  "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's",  "it is", text)
    text = re.sub(r"'s",  "", text)
        
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'d",  " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'re", " are", text)
        
    text = re.sub(r':\)', ' happy ', text)
    text = re.sub(r':d', ' happy ', text)
    text = re.sub(r':\(', ' sad ', text)
    text = re.sub(r':/', ' sad ', text)
    
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\!+', '!', text)
    text = re.sub(r'\?+', '?', text)

    filters=',-\(\)":;\'#$§%&*+<=>@[\\/]^_`{|}~\t\n'
    filters_dict = dict((i, " ") for i in filters)
    translations = str.maketrans(filters_dict)
    text = text.translate(translations)
    
    text = re.sub(' +', ' ', text)
    text = ''.join(text)

    return text