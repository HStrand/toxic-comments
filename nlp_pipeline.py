import pandas as pd
import numpy as np
import os
import re
import string
import gensim
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

"""
Feature engineering
"""
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments.apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)

pretrained = "data\\GoogleNews-vectors-negative300.bin.gz"

def w2v(series):
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(pretrained, binary=True)        
    return get_word2vec_embeddings(word_vectors, series)

def lengths(series):
    return np.array(series.apply(len)).reshape(-1,1).astype(float)

def asterixes(series):
    return np.array(series.apply(lambda x: x.count('!'))).reshape(-1,1).astype(float)

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

"""
Transforms
"""

def tokenize(row):
    return re_tok.sub(r' \1 ', row).lower().split()

"""
Pipeline class
"""
class NlpPipeline():

    def __init__(self, train=None, test=None, input_column='text_comment', class_labels=None, feature_functions=None, transforms=None, models=None, metric='roc_auc', id_column='id', verbosity=1):
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

    def run(self):        
        self.apply_transforms()
        self.engineer_features()
        self.train_test()
        # self.cross_val()
        self.fit_predict()
        self.create_submission()

    def log(self, s):
        if self.verbosity > 0:
            print(s)

    def engineer_features(self, use_transform=True, normalize=False):
        self.log("Engineering features")
        train_feats = []
        test_feats = []
        if use_transform:
            train_data = self.train_transformed
            test_data = self.test_transformed
        else:
            train_data = self.train[input_column]
            test_data = self.test[input_column]
            
        for func in self.feature_functions:
            train_feature = np.array(func(train_data))
            train_feats.append(train_feature)
            test_feature = np.array(func(test_data))
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
            train_data = self.train[input_column]
            test_data = self.test[input_column]
        
        train_feature = np.array(func(train_data))
        test_feature = np.array(func(test_data))
        if normalize:
            train_feature = self.normalize(train_feature)
            test_feature = self.normalize(test_feature)
        
        self.train_features = np.hstack((self.train_features, np.array(train_feature)))
        self.test_features = np.hstack((self.test_features, np.array(test_feature)))
        
    def normalize(self, data):
        self.scaler.fit(data)
        return self.scaler.transform(data)
    
    def apply_transforms(self):
        self.train_transformed = self.train[input_column]
        self.test_transformed = self.test[input_column]
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
                scores = cross_val_score(model, self.train_features, list(train[label]), scoring=self.metric, cv=5)
                self.log(self.metric + str(np.mean(scores)))
                scorelist.append(np.mean(scores))
            self.cv_scores[model.name] = np.mean(scorelist)

    def fit_predict(self):
        self.log("Fitting and predicting") 
        self.predictions = {}
        for model in self.models:
            self.predictions[model.name] = {}
            for label in self.class_labels:
                self.log("Fitting submission classifier for " + label)
                y_train = np.array(train[label])
                model.fit(self.train_features, y_train)
                self.predictions[model.name][label] = model.predict_proba(self.test_features)

    def create_submission(self):
        for model in self.models:
            self.log("Creating submissions")
            submission = self.test[self.id_column].to_frame()
            for label in self.class_labels:
                submission[label] = self.predictions[model.name][label][:,1]
            
            submission_num = 1
            past_submissions = self.get_past_submissions()
            if past_submissions is not None and past_submissions != []:
                submission_num = max(past_submissions) + 1
            filename = 'submissions\\submission' + str(submission_num) + '.csv'
            submission.to_csv(filename, index=False)
            self.store_submission_metadata(filename, submission_num, model)

    def get_past_submissions(self):
        current_dir = os.getcwd()
        path = os.path.join(current_dir, 'submissions')
        try:
            return [int(s[s.find('.csv')-1]) for s in os.listdir(path) if s.find('.csv') > -1 and s.find('submission') > -1]
        except:
            return None

    def store_submission_metadata(self, filename, submission_num, model):
        feature_funcs = ""
        transforms = ""
        for func in self.feature_functions:
            feature_funcs += str(func).split(' ')[1] + " "
        for trf in self.transforms:
            transforms += str(trf).split(' ')[1] + " "   
        cols = ["submission", "filename", "model", "feature_funcs", "transforms", "cv_score"]
        metadata = pd.DataFrame([[submission_num, filename, self.model_info(model), feature_funcs, transforms, self.cv_scores[model.name]]], columns=cols)
        filename = 'submissions\\submeta.csv'
        try:
            df = pd.read_csv(filename)
            metadata.to_csv(filename, mode='a', header=False, index=False)
        except:            
            metadata.to_csv(filename, mode='a', index=False)
            
    def model_info(self, model):
        s = model.name + ":"
        for param in model.get_params():            
            s += " "
            s += str(model.get_params()[param])
        
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

    def __init__(self, train=None, test=None, input_column='text_comment', class_labels=None, feature_functions=None, transforms=None, models=None, metric='roc_auc', id_column='id', verbosity=1):
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

    def run(self):        
        self.apply_transforms()
        self.engineer_features()
        self.train_test()
        # self.cross_val()
        self.fit_predict()
        self.create_submission()

    def log(self, s):
        if self.verbosity > 0:
            print(s)

    def engineer_features(self, use_transform=True, normalize=False):
        self.log("Engineering features")
        train_feats = []
        test_feats = []
        if use_transform:
            train_data = self.train_transformed
            test_data = self.test_transformed
        else:
            train_data = self.train[input_column]
            test_data = self.test[input_column]
            
        for func in self.feature_functions:
            train_feature = np.array(func(train_data))
            train_feats.append(train_feature)
            test_feature = np.array(func(test_data))
            test_feats.append(test_feature)

        self.train_features = np.hstack((feature for feature in train_feats))
        self.test_features = np.hstack((feature for feature in test_feats))
        
    def add_feature(self, func, use_transform=False, normalize=True):
        self.log("Adding feature")
        if use_transform:
            train_data = self.train_transformed
            test_data = self.test_transformed
        else:
            train_data = self.train[input_column]
            test_data = self.test[input_column]
        
        train_feature = np.array(func(train_data))
        test_feature = np.array(func(test_data))
        if normalize:
            train_feature = self.normalize(train_feature)
            test_feature = self.normalize(test_feature)
        
        self.train_features = np.hstack((self.train_features, np.array(train_feature)))
        self.test_features = np.hstack((self.test_features, np.array(test_feature)))
        
    def normalize(self, data):
        self.scaler.fit(data)
        return self.scaler.transform(data)
    
    def apply_transforms(self):
        self.train_transformed = self.train[input_column]
        self.test_transformed = self.test[input_column]
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
                scores = cross_val_score(model, self.train_features, list(train[label]), scoring=self.metric, cv=5)
                self.log(self.metric + str(np.mean(scores)))
                scorelist.append(np.mean(scores))
            self.cv_scores[model.name] = np.mean(scorelist)

    def fit_predict(self):
        self.log("Fitting and predicting") 
        self.predictions = {}
        for model in self.models:
            self.predictions[model.name] = {}
            for label in self.class_labels:
                self.log("Fitting submission classifier for " + label)
                y_train = np.array(train[label])
                model.fit(self.train_features, y_train)
                self.predictions[model.name][label] = model.predict_proba(self.test_features)

    def create_submission(self):
        for model in self.models:
            self.log("Creating submissions")
            submission = self.test[self.id_column].to_frame()
            for label in self.class_labels:
                submission[label] = self.predictions[model.name][label][:,1]
            
            submission_num = 1
            past_submissions = self.get_past_submissions()
            if past_submissions is not None and past_submissions != []:
                submission_num = max(past_submissions) + 1
            filename = 'submissions\\submission' + str(submission_num) + '.csv'
            submission.to_csv(filename, index=False)
            self.store_submission_metadata(filename, submission_num, model)

    def get_past_submissions(self):
        current_dir = os.getcwd()
        path = os.path.join(current_dir, 'submissions')
        try:
            return [int(s[s.find('.csv')-1]) for s in os.listdir(path) if s.find('.csv') > -1 and s.find('submission') > -1]
        except:
            return None

    def store_submission_metadata(self, filename, submission_num, model):
        feature_funcs = ""
        transforms = ""
        for func in self.feature_functions:
            feature_funcs += str(func).split(' ')[1] + " "
        for trf in self.transforms:
            transforms += str(trf).split(' ')[1] + " "   
        cols = ["submission", "filename", "model", "feature_funcs", "transforms", "cv_score"]
        metadata = pd.DataFrame([[submission_num, filename, self.model_info(model), feature_funcs, transforms, self.cv_scores[model.name]]], columns=cols)
        filename = 'submissions\\submeta.csv'
        try:
            df = pd.read_csv(filename)
            metadata.to_csv(filename, mode='a', header=False, index=False)
        except:            
            metadata.to_csv(filename, mode='a', index=False)
            
    def model_info(self, model):
        s = model.name + ":"
        for param in model.get_params():            
            s += " "
            s += str(model.get_params()[param])
        
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


if __name__ == "__main__":
    
    train = pd.read_csv('data\\train.csv')
    test = pd.read_csv('data\\test.csv')

    # Pipeline inputs
    input_column = 'comment_text'
    class_labels = [column for column in train.columns[2:8]]
    feature_funcs = [w2v]
    transforms = [tokenize]
    logreg = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg')
    logreg.name = "Logistic regression newton"
    models = [logreg]

    pipeline = NlpPipeline(train, test, input_column, class_labels, feature_funcs, transforms, models)
    pipeline.run()