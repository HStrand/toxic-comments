import pandas as pd
import numpy as np
import os
import re
import string
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb

"""
Pretrained model
"""

def get_average_wordvector(tokens_list, vector, generate_missing=False, dim=300):
    if len(tokens_list)<1:
        return np.zeros(dim)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(dim) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(dim) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments.apply(lambda x: get_average_wordvector(x, vectors, generate_missing=generate_missing))
    return list(embeddings)

def get_coefs(row, dim=300):
    row = row.strip().split()
    word, arr = " ".join(row[:-dim]), row[-dim:]
    return word, np.asarray(arr, dtype='float32')

def get_pretrained(text_file):
    return dict(get_coefs(row) for row in open(text_file, encoding="utf-8"))

"""
Feature engineering
"""
def format_feature(series, func):
    return np.array(series.apply(func)).reshape(-1,1).astype(float)

def asterix_freq(x):
    return x.count('!')/len(x)

def question_freq(x):
    return x.count('?')/len(x)    

def uppercase_freq(x):
    return len(re.findall(r'[A-Z]',x))/len(x)

def line_change_freq(x):
    return x.count("\n")/len(x)

def rep_freq(x):
    return np.sum([x[i]==x[i+1]==x[i+2] for i in range(0,len(x)-2)])/len(x) 

def has_ip(x):
    ips_in_text = re.findall( r'[0-9]+(?:\.[0-9]+){3}', x)
    has_ip = len(ips_in_text) > 0
    return int(has_ip)

def has_talk_tag(x):
    return int("(talk" in x)

def has_utc(x):
    return int("(UTC)" in x)

def link_count(x):
    return x.count("http")

def starts_with_i(x):
    return int(x[:2] == "I ")

def starts_with_you(x):
    return x.lower().startswith("you")

def about_image(x):
    return int("Image:" in x)

"""
Transforms
"""
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(row):
    return re_tok.sub(r' \1 ', row).lower().split()

"""
Pipeline class
"""
class NlpPipeline():

    def __init__(self, train=None, test=None, input_column='text_comment', class_labels=None, feature_functions=None, transforms=None, models=None, metric='roc_auc', word_index=None, pretrained=None, id_column='id', verbosity=1):
        self.train = train
        self.test = test
        self.input_column = input_column
        self.class_labels = class_labels        
        self.feature_functions = feature_functions
        self.transforms = transforms
        self.models = models
        self.metric = metric
        self.id_column = id_column
        self.verbosity = verbosity
        self.train_features = np.array([])
        self.test_features = np.array([])
        self.cv_scores = {}
        for model in self.models:
            self.cv_scores[model.name] = -1
        self.scaler = StandardScaler()
        self.train_transformed = train[input_column]
        self.test_transformed = test[input_column]
        self.word_index = word_index
        self.pretrained = pretrained
        self.oof_preds = {}

    def run(self):        
        self.engineer_features()
        self.apply_transforms()
        self.create_embeddings()
        self.fit_predict_oof()
        self.create_submission()

    def log(self, s):
        if self.verbosity > 0:
            print(s)

    def create_embeddings(self):
        self.log("Creating embeddings")
        embeddings = get_embeddings(self.word_index, self.train_transformed.append(self.test_transformed))

        if self.train_features.size == 0:
            self.train_features = np.array(embeddings[:len(self.train[self.input_column])])
        else:
            self.train_features = np.hstack((self.train_features, np.array(embeddings[:len(self.train[self.input_column])])))
        if self.test_features.size == 0:
            self.test_features = np.array(embeddings[len(self.test[self.input_column]):])
        else:
            self.test_features = np.hstack((self.test_features, np.array(embeddings[len(self.train[self.input_column]):])))

    def engineer_features(self, use_transform=False, normalize=True):
        self.log("Engineering features")
        train_feats = []
        test_feats = []
        if use_transform:
            train_data = self.train_transformed
            test_data = self.test_transformed
        else:
            train_data = self.train[self.input_column]
            test_data = self.test[self.input_column]
            
        for func in self.feature_functions:
            train_feature = format_feature(train_data, func)
            test_feature = format_feature(test_data, func)
            if normalize:
                train_feature = self.normalize(train_feature)
                test_feature = self.normalize(test_feature)
            train_feats.append(train_feature)
            test_feats.append(test_feature)

        self.train_features = np.hstack((feature for feature in train_feats))
        self.test_features = np.hstack((feature for feature in test_feats))
        
    def add_feature(self, func, use_transform=False, normalize=True):
        self.log("Adding feature")
        self.feature_functions.append(func)
        if use_transform:
            train_data = self.train_transformed
            test_data = self.test_transformed
        else:
            train_data = self.train[self.input_column]
            test_data = self.test[self.input_column]
        
        train_feature = format_feature(train_data, func)
        test_feature = format_feature(test_data, func)
        if normalize:
            train_feature = self.normalize(train_feature)
            test_feature = self.normalize(test_feature)
        
        self.train_features = np.hstack((self.train_features, train_feature))
        self.test_features = np.hstack((self.test_features, test_feature))

    def add_feature_data(self, train_feature, test_feature, normalize=False):
        if normalize:
            train_feature = self.normalize(train_feature)
            test_feature = self.normalize(test_feature)
        self.train_features = np.hstack((self.train_features, train_feature))
        self.test_features = np.hstack((self.test_features, test_feature))

        
    def normalize(self, data):
        self.scaler.fit(data)
        return self.scaler.transform(data)
    
    def apply_transforms(self):
        self.train_transformed = self.train[self.input_column]
        self.test_transformed = self.test[self.input_column]
        self.log("Applying transforms")
        for transform in self.transforms:
            self.train_transformed = self.train[self.input_column].apply(transform)
            self.test_transformed = self.test[self.input_column].apply(transform)

    def train_test(self):
        self.log("Training and testing") 
        for model in self.models:
            self.log(str(model)) 
            scorelist = [] 
            for label in self.class_labels:
                self.log("Fitting classifier for " + label)
                X_train, X_test, y_train, y_test = train_test_split(self.train_features, list(self.train[label]), test_size=0.2, random_state=40)
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_test)
                self.log(self.metric + str(roc_auc_score(y_test, y_pred[:,1])))
                scorelist.append(np.mean(roc_auc_score(y_test, y_pred[:,1])))
            self.cv_scores[model.name] = np.mean(scorelist)

    def cross_val(self):
        self.log("Cross-validating") 
        for model in self.models:
            self.log(str(model)) 
            scorelist = [] 
            for label in self.class_labels:
                self.log("Cross-validating " + label)
                scores = cross_val_score(model, self.train_features, list(self.train[label]), scoring=self.metric, cv=5)
                self.log(self.metric + ": " + str(np.mean(scores)))
                scorelist.append(np.mean(scores))
            self.cv_scores[model.name] = np.mean(scorelist)

    def fit_predict(self):
        self.log("Fitting and predicting") 
        self.predictions = {}
        for model in self.models:
            self.predictions[model.name] = {}
            for label in self.class_labels:
                self.log("Fitting submission classifier for " + label)
                y_train = np.array(self.train[label])
                model.fit(self.train_features, y_train)
                self.predictions[model.name][label] = model.predict_proba(self.test_features)

    """
    Fit each model on 4 folds and predict the 5th fold.
    These predictions are stored in self.oof_preds and saved to be used to fit stackers.
    Finally, fit on the entire data set and predict on test set.
    These predictions are stored in self.predictions and saved to be used as test inputs for stackers.
    """
    def fit_predict_oof(self):
        self.log("Creating out-of-fold meta training set for stacker")
        folds = KFold(n_splits=5, shuffle=True, random_state=42)
        self.oof_preds = {}
        for model in self.models:
            scorelist = []
            self.oof_preds[model.name] = {}
            self.log(str(model))
            for label in self.class_labels:
                self.log(label)
                self.oof_preds[model.name][label] = np.zeros((self.train_features.shape[0]))
                for train_idx, pred_idx in folds.split(self.train_features):
                    X_train = self.train_features[train_idx]
                    y_train = self.train[label][train_idx]
                    X_holdout = self.train_features[pred_idx]
                    y_holdout = self.train[label][pred_idx]
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_holdout)[:,1]
                    self.oof_preds[model.name][label][pred_idx] = y_pred
                    auc = roc_auc_score(y_holdout, y_pred)
                    self.log("AUC: " + str(auc))
                    scorelist.append(auc)
            self.cv_scores[model.name] = np.mean(scorelist)
            self.log("CV score: " + str(self.cv_scores[model.name]))

        self.fit_predict() # Finally, fit on entire training set and predict

    def create_submission(self, oof=True):
        for model in self.models:
            self.log("Creating submissions")
            submission = self.test[self.id_column].to_frame()
            for label in self.class_labels:
                submission[label] = self.predictions[model.name][label][:,1]
            
            submission_num = 1
            past_submissions = self.get_past_submissions()
            if past_submissions is not None and past_submissions != []:
                submission_num = max(past_submissions)[0] + 1
            filename = 'submissions\\submission' + str(submission_num) + '.csv'
            submission.to_csv(filename, index=False)
            
            if oof:
                oof_name = "submissions\\oof_train" + str(submission_num) + ".csv"
                oof = self.train[self.id_column].to_frame()
                for label in self.class_labels:
                    oof[label] = self.oof_preds[model.name][label]
                oof.to_csv(oof_name, index=False)
            
            self.store_submission_metadata(filename, submission_num, model)

    def get_past_submissions(self):
        current_dir = os.getcwd()
        path = os.path.join(current_dir, 'submissions')
        try:
            return [[int(s) for s in re.findall(r'\d+', f)] for f in os.listdir(path)]
        except:
            return None

    def store_submission_metadata(self, filename, submission_num, model):
        feature_funcs = ""
        transforms = ""
        for func in self.feature_functions:
            feature_funcs += str(func).split(' ')[1] + " "
        for trf in self.transforms:
            transforms += str(trf).split(' ')[1] + " "   
        cols = ["submission", "filename", "model", "pretrained", "feature_funcs", "transforms", "cv_score"]
        metadata = pd.DataFrame([[submission_num, filename, self.model_info(model), self.pretrained, feature_funcs, transforms, self.cv_scores[model.name]]], columns=cols)
        filename = 'submissions\\submeta.csv'
        try:
            df = pd.read_csv(filename)
            metadata.to_csv(filename, mode='a', header=False, index=False)
        except:            
            metadata.to_csv(filename, mode='a', header=False, index=False)
            
    def model_info(self, model):
        s = model.name + ":"
        try:
            for param in model.get_params():            
                s += " "
                s += str(model.get_params()[param])
        except:
            pass
        
        return s
    
    def __repr__(self):
        s = "Train: "
        s += str(self.train.shape)
        s += "\n"
        s += "Test: "
        s += str(self.test.shape)
        s += "\n"
        s += "Train features: "
        s += str(self.train_features.shape)
        s += "\n"
        s += "Test features: "
        s += str(self.test_features.shape)
        s += "\n"
        s += "Input column: "
        s += self.input_column
        s += "\n"
        s += "Class labels:"
        for label in self.class_labels:
            s += " "
            s += label
        s += "\n"
        s += "Models: "
        for model in self.models:            
            s += self.model_info(model)
            s += " | "            
        s += "\n"
        s += "Transforms: "
        for transform in self.transforms:
            s += " "
            s += str(transform).split(' ')[1]
        s += "\n"
        s += "Feature functions: "
        for func in self.feature_functions:
            s += " "
            s += str(func).split(' ')[1]
        s += "\n"
        s += "Metric: "
        s += self.metric
        s += "\n"
        s += "CV scores: "
        s += str(self.cv_scores)
        
        return s

"""
Main
"""
if __name__ == "__main__":
    
    train = pd.read_csv('data\\train.csv').fillna(' ')
    test = pd.read_csv('data\\test.csv').fillna(' ')

    # pretrained = "data\\GoogleNews-vectors-negative300.bin.gz"
    # pretrained = 'data\\glove.840B.300d.txt'
    # pretrained = 'data\\glove.6B.300d.txt'
    pretrained = "data\\crawl-300d-2M.vec"

    # print("Getting google news model")
    # w2v = gensim.models.KeyedVectors.load_word2vec_format(pretrained, binary=True)   

    print("Getting pretrained model from", pretrained)
    word_index = get_pretrained(pretrained)

    # print("Getting google news model")
    # w2v = gensim.models.KeyedVectors.load_word2vec_format(pretrained, binary=True)   

    # Pipeline inputs
    class_labels = [column for column in train.columns[2:8]]
    feature_funcs = [len, asterix_freq, uppercase_freq, line_change_freq, rep_freq, question_freq, has_ip, has_talk_tag, link_count, starts_with_i, starts_with_you, about_image]
    transforms = [tokenize]
    logreg = LogisticRegression(C=0.2, class_weight='balanced', solver='newton-cg', max_iter=10)
    logreg.name = "Logistic regression newton"
    gbm = lgb.LGBMClassifier(metric="auc", num_leaves=31, boosting_type="gbdt", learning_rate=0.1, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5, reg_lambda=0.5)
    gbm.name = "LightGBM stacker"
    models = [gbm]

    pipe = NlpPipeline(train, test, "comment_text", class_labels, feature_funcs, transforms, models, word_index=word_index, pretrained=pretrained)
    print(pipe)
    pipe.run()