import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, PReLU, Add
from keras.layers import Bidirectional, GlobalMaxPool1D, BatchNormalization, SpatialDropout1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from attlayer import AttentionWeightedAverage
from nlp_pipeline import *

N_DIMS = 300

def get_coefs(row):
    row = row.strip().split()
    word, arr = " ".join(row[:-N_DIMS]), row[-N_DIMS:]
    return word, np.asarray(arr, dtype='float32')

def get_pretrained(text_file):
    return dict(get_coefs(row) for row in open(text_file, encoding="utf-8"))

def get_indices(fold):
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = [idx for idx in folds.split(train["id"])]
    train_idx = indices[fold][0]
    pred_idx = indices[fold][1]
    return train_idx, pred_idx  

def halve(epoch):
    base = 0.002
    return base/(2**epoch)

def decay07(epoch):
    base = 0.001
    return base*(0.7**epoch)

class LstmNet():
    
    def __init__(self, embed_size, max_features, maxlen, embedding_matrix, num_features):
        input1 = Input(shape=(maxlen,))
        model1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(input1)
        model1 = Bidirectional(LSTM(300, return_sequences=True))(model1)
        model1 = AttentionWeightedAverage()(model1)
        # model1 = GlobalMaxPool1D()(model1)
        model1 = Dense(300, activation="relu")(model1)
        model1 = Dropout(0.5)(model1)
        out = Dense(6, activation="sigmoid")(model1)
        self.model = Model(inputs=input1, outputs=out)
        self.model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    
    def fit(self, train_features, train_labels):
        # early = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=0, verbose=0, mode='auto')
        # file_path="weights_base.best.hdf5"
        # checkpoint = callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        lrate = LearningRateScheduler(decay07)
        self.model.fit(train_features, y=train_labels, batch_size=32, epochs=3, validation_split=0.1, shuffle=True, callbacks=[lrate])
        # self.model.fit(train_features, train_labels, batch_size=32, epochs=2)

    def predict_proba(self, X):
        self.predictions = self.model.predict(X, batch_size=1024, verbose=1)
        return self.predictions

    def submit(self):
        sub = pd.read_csv('data\\sample_submission.csv')
        sub[list_classes] = self.predictions
        sub.to_csv('submissions\\lstm17.csv', index=False)

if __name__ == "__main__":

    train = pd.read_csv('data\\train.csv').fillna(' ')
    test = pd.read_csv('data\\test.csv').fillna(' ')

    embed_size = 300
    max_features = 394787
    maxlen = 500
    num_features = 12

    list_sentences_train = train["comment_text"].values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    list_sentences_test = test["comment_text"].values

    print("Tokenizing")
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(np.concatenate([list_sentences_train, list_sentences_test])))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

    pretrained = "data\\crawl-300d-2M.vec"
    print("Getting", pretrained)
    embeddings_index = get_pretrained(pretrained)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    oof = False

    if oof:
        fold = 0
        train_idx, pred_idx = get_indices(fold)
        net = LstmNet(embed_size, max_features, maxlen, embedding_matrix)
        net.fit(X_t[train_idx], y[train_idx])
        y_oof = net.predict_proba(X_t[pred_idx])
        
        sub_oof = pd.read_csv('submissions\\lstm_ft_oof_template.csv', encoding="utf-8")
        for i in range(0,len(list_classes)):
            sub_oof[list_classes[i]][pred_idx] = y_oof[:,i]
        sub_oof.to_csv('lstm_ft_oof_template.csv', index=False, encoding="utf-8")
    
    else:
        net = LstmNet(embed_size, max_features, maxlen, embedding_matrix, num_features)
        net.fit(X_t, y)
        # file_path="weights_base.best.hdf5"
        # net.model.load_weights(file_path)
        y_test = net.predict_proba(X_te)
        net.submit()