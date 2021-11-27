def main():
    print('start ...')
    
    print('reuters')

    import logging
    import os 
    import sys
    import numpy as np
    from tqdm import tqdm
    import tensorflow.keras as tfk    

    cwd = os.getcwd()
    sys.path.append(cwd + '/../.') 

    from src.util.data_loader import Dataset, load_data
    from src.util.sbert_transformer import SbertSentenceMeanEncoder, SbertFullTextEncoder
    from src.model.stats import predict, print_all_stats
    from src.model.neural_networks import MLP

    # load data
    X, y = load_data(Dataset.REUTERS)
    X = None

    # load encoding
    X_enc = np.load('./encodings/reuters_embeddings_full.npy')

    X_enc = np.array(X_enc)
    y = np.array(y)

    print('Naive Bayes')
    from sklearn.naive_bayes import GaussianNB
    cls = GaussianNB
    dfs = predict(cls, (X_enc), y, skl_model=True)
    print_all_stats(dfs, 'reuters_gnb')

    print('Logistic Regression')
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression
    model_args = {'random_state': 42, 'max_iter': 100}
    dfs = predict(clf, X_enc, y, skl_model=True, **model_args)
    print_all_stats(dfs, 'reuters_lr')

    print('Random Forest Classifier')
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier
    model_args = {'n_estimators': 100, 'bootstrap': True, 'random_state': 42}
    dfs = predict(clf, X_enc, y, skl_model=True, **model_args)
    print_all_stats(dfs, 'reuters_rf')

    from sklearn.neighbors import KNeighborsClassifier
    print('K-Nearest Neighbors')
    clf = KNeighborsClassifier
    model_args = {'n_neighbors': 25}
    dfs = predict(clf, X_enc, y, skl_model=True, **model_args)
    print_all_stats(dfs, 'reuters_knn')

    print('Suport Vector Machine')
    from sklearn import svm
    from src.model.wrapper import SklearnWrapper
    clf = SklearnWrapper
    model_args = {'Clf' : svm.SVC, 'kernel': 'linear', 'probability': True}
    dfs = predict(clf, X_enc, y, skl_model=True, **model_args)
    print_all_stats(dfs, 'reuters_svm')

    print('Multi Layer Perceptron (TF)')
    callback = tfk.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001)
    model_args = {"input_shape": 768, "n_classes": 8, 'mcd':False, 'T':1, 'callbacks': [callback]}
    dfs = predict(MLP, X_enc, y, batch_size=128, epochs=100, val=True, save='reuters', **model_args)
    print_all_stats(dfs, 'reuters_mlp_tf')

    print('Bayesian Multi Layer Perceptron (TF)')
    callback = tfk.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001)
    model_args = {"input_shape": 768, "n_classes": 8, 'mcd':True, 'T':100, 'callbacks': [callback]}
    dfs = predict(MLP, X_enc, y, batch_size=128, epochs=100, val=True, load='reuters', **model_args)
    print_all_stats(dfs, 'reuters_bmlp_tf')

    print('Multi Layer Perceptron (Sklearn)')
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier
    model_args = {'random_state': 42, 'early_stopping': True, 'max_iter': 100, 'hidden_layer_sizes':(500, 500)}
    dfs = predict(clf, X_enc, y, skl_model=True, **model_args)
    print_all_stats(dfs, 'reuters_mlp')

    print('Decision Tree Classifier')
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier
    model_args = {'random_state': 42}
    dfs = predict(clf, X_enc, y, skl_model=True, **model_args)
    print_all_stats(dfs, 'reuters_dt')

    print('... end')

if __name__ == "__main__":
    main()