from pymongo import MongoClient
import random
from datetime import datetime

import nltk
from nltk import word_tokenize, NaiveBayesClassifier, classify
nltk.download('punkt')

from .Quote import Quote
from .TrainingResult import TrainingResult

# Naive Bayes classifier
class Engine:
    def __init__(self, db, display):
        self._db = db
        self._clf_dict = {}
        self._display = display

    # Predict the rating of a quote
    def test_quote(self, quote: Quote, user_id: str) -> bool:
        if (user_id not in self._clf_dict):
            print('no classifier exists for the user!')
            return False
        
        features = self._get_features(quote.quote_text)
        res = self._clf_dict[user_id].classify(features)
        return res == '1'

    # Train the classifier
    def train(self, user_id: str) -> TrainingResult:
        # Get all docs from mongo and split to pos/neg
        all_docs = list(self._db.find({'user_id': user_id}))
        pos_docs = [x for x in all_docs if x['rating'] == 1]
        neg_docs = [x for x in all_docs if x['rating'] == -1]
        
        self._my_print(f'# positive ratings: {len(pos_docs)}')
        self._my_print(f'# negative ratings: {len(neg_docs)}')

        # Build the labeled set and shuffle
        all_quotes = [(q['quote']['quote_text'], '1') for q in pos_docs]
        all_quotes += [(q['quote']['quote_text'], '-1') for q in neg_docs]
        random.seed(datetime.now())
        random.shuffle(all_quotes)

        # Construct the features
        all_features = [(self._get_features(quote), label)
            for (quote, label) in all_quotes]

        # Train it and create the classifier
        train_size = int(len(all_features) * 0.8)
        train_set, test_set = all_features[:train_size], all_features[train_size:]
        self._my_print(f"Training set size = {str(len(train_set))} quotes")
        self._my_print(f"Test set size = {str(len(test_set))} quotes")
        clf = NaiveBayesClassifier.train(train_set)

        # Results
        test_accuracy = str(classify.accuracy(clf, test_set))
        
        self._my_print(f"Accuracy on the training set = {str(classify.accuracy(clf, train_set))}")
        self._my_print(f"Accuracy of the test set = {test_accuracy}")
        if (self._display):
            clf.show_most_informative_features(50)

        self._clf_dict[user_id] = clf
        return TrainingResult(test_accuracy, len(train_set))  

     
    def _get_features(self, text):
        features = {}
        word_list = [word for word in word_tokenize(text.lower())]
        for word in word_list:
            features[word] = True
        return features
    
    def _my_print(self, text):
        if (self._display):
            print(text)

