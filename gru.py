import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from attlayer import AttentionWeightedAverage

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

class GruNet():
    
    def __init__(self, embed_size, max_features, maxlen, embedding_matrix):
        inp = Input(shape=(maxlen,))
        x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
        x = Bidirectional(GRU(300, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
        # x = GlobalMaxPool1D()(x)
        x = AttentionWeightedAverage()(x)
        x = Dense(300, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(6, activation="sigmoid")(x)
        self.model = Model(inputs=inp, outputs=x)
        # optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
        self.model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    
    def fit(self, train_features, train_labels):
        # early = callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=0, verbose=0, mode='auto')
        # file_path="weights_base.best.hdf5"
        # checkpoint = callbacks.ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True, mode='min')
        self.model.fit(train_features, train_labels, batch_size=32, epochs=2, validation_split=0.1)
        # self.model.fit(train_features, train_labels, batch_size=32, epochs=2)

    def predict_proba(self, features):
        self.predictions = self.model.predict([features], batch_size=1024, verbose=1)
        return self.predictions

    def submit(self):
        sub = pd.read_csv('data\\sample_submission.csv')
        sub[list_classes] = self.predictions
        sub.to_csv('submissions\\gru2.csv', index=False)


if __name__ == "__main__":

    train = pd.read_csv('data\\train.csv').fillna(' ')
    test = pd.read_csv('data\\test.csv').fillna(' ')

    embed_size = 300
    max_features = 394787
    maxlen = 100

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

    oof = True

    if oof:
        fold = 0
        train_idx, pred_idx = get_indices(fold)
        net = GruNet(embed_size, max_features, maxlen, embedding_matrix)
        net.fit(X_t[train_idx], y[train_idx])
        y_oof = net.predict_proba(X_t[pred_idx])
        
        sub_oof = pd.read_csv('submissions\\gru_ft_oof_template.csv', encoding="utf-8")
        for i in range(0,len(list_classes)):
            sub_oof[list_classes[i]][pred_idx] = y_oof[:,i]
        sub_oof.to_csv('gru_ft_oof_template.csv', index=False, encoding="utf-8")
    
    else:
        net = GruNet(embed_size, max_features, maxlen, embedding_matrix)
        net.fit(X_t, y)
        # file_path="weights_base.best.hdf5"
        # net.model.load_weights(file_path)
        y_test = net.predict_proba(X_te)
        net.submit()