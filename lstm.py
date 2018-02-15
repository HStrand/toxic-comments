import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use

class LstmNet():
    
    def __init__(self, embed_size, max_features, maxlen):
        inp = Input(shape=(maxlen,))
        x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
        x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(6, activation="sigmoid")(x)
        self.model = Model(inputs=inp, outputs=x)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def fit(self, train_features, train_label):
        early = [callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')]
        self.fit(train_features, train_label, batch_size=64, epochs=5, validation_split=0.1, callbacks=early)

    def predict_proba(features):
        self.predictions = self.model.predict([features], batch_size=1024, verbose=1)
        return self.predictions

    def submit():
        sub = pd.read_csv('C:\\Code\\Kaggle\\toxic-comments\\data\\sample_submission.csv')
        sub[list_classes] = self.predictions
        sub.to_csv('lstm_preds.csv', index=False)