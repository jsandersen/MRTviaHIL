import os 
import sys
import numpy as np
from tqdm import tqdm

cwd = os.getcwd()
sys.path.append(cwd + '/../../.') 

from src.util.data_loader import Dataset, load_data
from src.util.sbert_transformer import SbertSentenceMeanEncoder, SbertFullTextEncoder
from src.util.preprocessing import preprocess_corpus

def main():
    X, _ = load_data(Dataset.IMDB, './../../aclImdb/')

    # Preprocessing
    X_clean = preprocess_corpus(X, keep_nbr=True)

    # encode mean encodings
    bert = SbertSentenceMeanEncoder()
    X_enc = [bert.encode(x) for x in tqdm(X_clean)]
    np.save('imdb_embeddings_mean.npy', X_enc)

if __name__ == "__main__":
    main()