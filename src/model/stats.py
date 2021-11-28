import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, roc_auc_score

from tensorflow.keras.utils import to_categorical


from timeit import default_timer as timer

from random import random, randint, seed

from tqdm import tqdm
import math

from scipy.stats import entropy

def _round_05(x):
    y = math.floor(x*10)/10
    if y + 0.05 <= x:
        y += 0.05
    return round(y, 3)

def brier_multi(targets, probs):
    return np.mean(np.sum((probs - targets)**2, axis=1))

def predict(Model, X, y, skl_model=False, batch_size=None, epochs=None, callbacks=None, val=False, prob=True, save=None, load=None, **kwargs):     
    
    dfs = []
    f1_micro_list = []
    f1_macro_list = []    
    auc_roc_list = []
    brier_list = []
    training_time = []
    inference_time = []
    it = 0
    
    sss = StratifiedShuffleSplit(n_splits=5,test_size=0.50, random_state=42) # 5
    
    for train_index, test_index in tqdm(sss.split(X, y), position=0, leave=True):
        
        clf = Model(**kwargs)
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if skl_model:
            train_start = timer()
            clf.fit(X_train, y_train)
            train_end = timer()
        else:
        # fit
        
            if load:
                train_end = -1
                train_start = 0
                clf.load_model('./model/%s_%s' % (load, it))
            else:
                if val:
                    train_start = timer()
                    clf.fit(X_train, y_train, X_test, y_test, batch_size, epochs, callbacks, 0.1)
                    train_end = timer()
                else :
                    train_start = timer()
                    clf.fit(X_train, y_train, batch_size, epochs, callbacks, 0)
                    train_end = timer()    
            if save:
                clf.save_model('./model/%s_%s' % (save, it))
        
        training_time.append(train_end - train_start)
        it = it + 1
        
        # infer
        if prob:
            inference_start = timer()
            y_prob = clf.predict_proba(X_test)
            inference_end = timer()
            inference_time.append(inference_end - inference_start)
            y_pred = y_prob.argmax(axis=1)
        else:
            inference_start = timer()
            uncertainty = clf.uncertainty(X_test)
            inference_end = timer()
            if len(np.unique(y)) > 2:
                y_pred = uncertainty.argmax(axis=1)
                uncertainty = uncertainty.max(axis=1)
            else: 
                y_pred = [0 if i < 0 else 1 for i in uncertainty]
            inference_time.append(inference_end - inference_start)
    
        # evaluate 
        f1_micro_list.append(f1_score(y_test, y_pred, average='micro'))
        f1_macro_list.append(f1_score(y_test, y_pred, average='macro'))
        brier_list.append(brier_multi(np.array(to_categorical(y_test)), np.array(y_prob)))
        
        if prob:
            if len(np.unique(y)) == 2:
                auc_roc_list.append(roc_auc_score(y_test, y_prob[:,1]))
            else:
                auc_roc_list.append(roc_auc_score(y_test, y_prob, multi_class='ovr'))

        y_prob = [np.array(x) for x in y_prob]
        # save as dataframe
        data = {'y_pred': y_pred, 'y_prob': y_prob, 'y_true': y_test, 'unc' : entropy(y_prob, base=2, axis=1) if prob else uncertainty}
        df = pd.DataFrame(data=data)
        df_sort = df.sort_values('unc', ascending=prob).reset_index()
        
        dfs.append(df_sort)
        
        break
    
    f1_micro_list = np.array(f1_micro_list)
    f1_macro_list = np.array(f1_macro_list)
    brier_list = np.array(brier_list)
    auc_roc_list = np.array(auc_roc_list)
    
    training_time = np.array(training_time)
    inference_time = np.array(inference_time)
    print(' ')
    
    print('Model Performance (Mean/Std)')
    print('f1_micro: ', round(f1_micro_list.mean(), 4), round(f1_micro_list.std(), 4))
    print('f1_macro: ',  round(f1_macro_list.mean(), 4), round(f1_macro_list.std(), 4))
    print('auc_roc: ',  round(auc_roc_list.mean(), 4), round(auc_roc_list.std(), 4))
    
    print(' ')
    print('brier score: ',  round(brier_list.mean(), 4), round(brier_list.std(), 4))
    print(' ')
    
    print('Time (sec) (Mean/Std)')
    print('training: ',  round(training_time.mean(), 4), round(training_time.std(), 4))
    print('inference: ',  round(inference_time.mean(), 4), round(inference_time.std(), 4))
    
    print(' ')
    return dfs

def machine_f1(dfs):
    res = []
    del_rate = [0.0, 0.1, 0.2, 0.3]
    for i in del_rate:
        res_i = []
        for df in dfs:
            size = int(len(df['y_true'])*i)
            if i == 0:
                res_i.append((f1_score(df['y_true'].values, df['y_pred'].values,  average='macro')))
            else:
                res_i.append((f1_score(df['y_true'][:-size].values, df['y_pred'][:-size].values,  average='macro')))
        res.append(res_i)
        
    res = np.array(res).mean(axis=1)
    growth = [0]
    for i in range(1, len(res)):
        growth.append(( (res[i] - res[i-1]) / res[i-1]))
    
    print('Model Performance (deletion rate / macro-f1 / groth)')
    for i in range(len(res)):
        print("%s: %s (+%s) " % (del_rate[i], round(res[i], 4), round(growth[i], 4)) )
    print('')

def compute_moderation_effort(dfs, p_oracle = 1):
    print('p_oracle: ', p_oracle)
    y_gold = None
    
    if p_oracle < 1:
        y_gold = []
        for df in dfs:
            y_true = df['y_true'].values
            n_labels = len(set(y_true))

            human_labells = []
            for i in y_true:
                irand = random()
                if (irand > p_oracle):
                    human_labells.append(randint(0, n_labels-1))
                else:
                    human_labells.append(i)

            y_gold.append(human_labells)
    
    mod_efforts = []
    j = 0
    for df in dfs:
        mod_effort = []
        y_true = df['y_true'].values
        

        y_pred = df['y_pred'].values
        for i in tqdm(range(len(y_true)), position=0, leave=True):
            ai = y_pred[:len(y_true)-i]
            if not y_gold:
                human = y_true[len(ai):]
            else:
                human = y_gold[j][len(ai):]
            mod_y = np.concatenate([ai, human])
            f1 = f1_score(y_true, mod_y, average='macro')
            mod_effort.append(f1)
        mod_efforts.append(mod_effort)
        j = j +1
    return mod_efforts

def eval_moderation_effort(mod_efforts):
    fontsize = 20
    
    mod_effort_mean = np.array(mod_efforts).mean(axis=0)
    mod_effort_std = np.array(mod_efforts).std(axis=0)

    mod_effort = np.array(mod_efforts).mean(axis=0)
    plt.plot(mod_effort, label='Uncertainty')
    plt.plot([0, len(mod_effort)], [mod_effort[0], 1], 'black', linestyle='dashed', label='Random')
    plt.fill_between(range(len(mod_effort)), mod_effort_mean-mod_effort_std, mod_effort_mean+mod_effort_std, alpha=.3)
    plt.xlim((0, len(mod_effort)))
    plt.yticks(np.arange(_round_05(mod_effort_mean[0]), 1.05, 0.05), fontsize=fontsize)
    plt.ylabel('F1-Score', fontsize=fontsize)
    plt.xlabel('Moderation Effort', fontsize=fontsize)
    plt.xticks(np.arange(0, len(mod_effort)+1, len(mod_effort)/5), ['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=fontsize)
    plt.ylim((mod_effort[0], 1.0005))
    l = plt.legend(frameon=True, fontsize=fontsize, title="Moderation Strategy", fancybox=True)
    plt.setp(l.get_title(),fontsize=18)
    plt.show()

    print('Moderation Effort Needed')
    
    for i in [.81, .83, .85, .87, .89, .91, .93, .95, .97, .99]:
        try:
            print(i, 'f1 =>', round(np.where(mod_effort_mean>=i)[0][0]/len(mod_effort_mean), 4), 'effort')
        except IndexError:
            print(np.array(mod_effort_mean).max())
            break;
            
def print_all_stats(dfs, path):
    i = 0
    for df in dfs:
        df.to_pickle("./dfs/df_%s_%s.pkl" % (path, i))
        i = i + 1
    
    machine_f1(dfs)
    mod_efforts = compute_moderation_effort(dfs)
    np.save('./dfs/mod_100_%s' % path, mod_efforts)
    
    eval_moderation_effort(mod_efforts)
    mod_efforts = compute_moderation_effort(dfs, 0.95)
    np.save('./dfs/mod_95_%s' % path, mod_efforts)
    
    eval_moderation_effort(mod_efforts)   
    mod_efforts = compute_moderation_effort(dfs, 0.9)
    np.save('./dfs/mod_90_%s' % path, mod_efforts)
        
    eval_moderation_effort(mod_efforts)
    
    print(' ')
    print('#############################')
    print(' ')
    