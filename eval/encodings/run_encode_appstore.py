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
    X, _ = load_data(Dataset.APP_STORE, './../../dataset.csv')

    # Preprocessing
    X_clean = preprocess_corpus(X)

    # encode full encodings
    bert = SbertFullTextEncoder()
    X_enc = [bert.encode(x) for x in tqdm(X_clean)]
    np.save('appstore_embeddings_full.npy', X_enc)

if __name__ == "__main__":
    main()