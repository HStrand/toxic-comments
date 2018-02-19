import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score
import re, string
from sklearn.model_selection import KFold

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): 
    return re_tok.sub(r' \1 ', s).split()

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(x, y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


if __name__ == "__main__":
    train = pd.read_csv('data\\train.csv').fillna(' ')
    test = pd.read_csv('data\\test.csv').fillna(' ')
    subm = pd.read_csv('data\\sample_submission.csv')

    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    n = train.shape[0]
    print("Vectorizing")
    vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                smooth_idf=1, sublinear_tf=1 )
    trn_term_doc = vec.fit_transform(train["comment_text"])
    test_term_doc = vec.transform(test["comment_text"])

    x = trn_term_doc
    test_x = test_term_doc

    # preds = np.zeros((len(train), len(labels)))  

    # folds = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # for i, label in enumerate(labels):
    #     for train_idx, pred_idx in folds.split(train[label]):
    #         print('fit', label)
    #         m,r = get_mdl(x[train_idx], train[label][train_idx])
    #         preds[:,i][pred_idx] = m.predict_proba(x[pred_idx].multiply(r))[:,1]
    #         print(roc_auc_score(train[label][pred_idx], preds[:,i][pred_idx]))

    # print("Saving out-of-fold")
    # submid = pd.DataFrame({'id': subm["id"]})
    # submission = pd.concat([submid, pd.DataFrame(preds, columns = labels)], axis=1)
    # submission.to_csv('oof_train_nblogreg.csv', index=False)

    preds = np.zeros((len(test), len(labels)))  

    for i, label in enumerate(labels):
        print('fit', label)
        m,r = get_mdl(x, train[label])
        preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]

    print("Saving submission")
    final_submid = pd.DataFrame({'id': subm["id"]})
    final_submission = pd.concat([final_submid, pd.DataFrame(preds, columns = labels)], axis=1)
    final_submission.to_csv('nblogreg_preds.csv', index=False)