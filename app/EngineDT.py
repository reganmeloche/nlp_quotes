import random
from datetime import datetime
import numpy as np
from pymongo import MongoClient
from collections import Counter
import operator

import nltk
from nltk import word_tokenize, NaiveBayesClassifier, classify
nltk.download('punkt')

import sklearn
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')

from .Quote import Quote
from .TrainingResult import TrainingResult
from .DecisionTreeObj import DecisionTreeObj

pos_list = ["C", "D", "E", "F", "I", "J", "M", "N", "P", "R", "T", "U", "V", "W"]

# Decision Tree Classifier
class EngineDT:
    def __init__(self, db, display):
        self._db = db
        self._display = display
        self._clf_dict = {}

    def test_quote(self, quote: Quote, user_id: str) -> bool:
        if (user_id not in self._clf_dict):
            print('no classifier exists!')
            return False

        pred_set = [(quote.quote_text.split(' '), '1')]
        pred_docs = EngineDT.preprocess(pred_set)
        dt_obj = self._clf_dict[user_id]
        pred_data, _ = self._initialize_dataset(pred_set, pred_docs, dt_obj.selected_suffixes, dt_obj.unique_vocab)
        
        res = dt_obj.clf.predict(pred_data)
        return bool(res[0] == 1)


    def train(self, user_id: str) -> TrainingResult:
        ### RETRIEVE, PARSE, SHUFFLE DOCS ###
        all_docs = list(self._db.find({'user_id': user_id}))
        all_quotes = [(q['quote']['quote_text'].split(' '), q['rating']) for q in all_docs]
        random.seed(datetime.now())
        random.shuffle(all_quotes)
        
        ### MAKE A STRATIFIED SPLIT ###
        # train_set should have 80% of the data, which should have the same split as the original data
        # test_set should have 20% of the data, and also the same split
        values = [rating for (quote, rating) in all_quotes]
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_set = []
        test_set = []
        for train_index, test_index in split.split(all_quotes, values):
            train_set = [all_quotes[index] for index in train_index]
            test_set = [all_quotes[index] for index in test_index]
        
        # print(len([x for x in test_set if x[1] == '1'])/len(test_set))
        # print(len([x for x in train_set if x[1] == '1'])/len(train_set))
        
        ### SPACY PREPROCESSING ###
        train_docs = EngineDT.preprocess(train_set)
        test_docs = EngineDT.preprocess(test_set)

        ### SET UP FEATURES ###
        selected_suffixes = EngineDT.select_suffixes(train_docs, 0.3)
        self._my_print(f'SELECTED SUFFIX COUNT: {len(selected_suffixes)}')
        
        unique_vocab = EngineDT.unique_vocabulary(train_set, 1, -1, 0.3)
        self._my_print(f'UNIQUE WORD COUNT: {len(unique_vocab)}')

        train_data, train_targets = self._initialize_dataset(train_set, train_docs, selected_suffixes, unique_vocab)
        test_data, test_targets = self._initialize_dataset(test_set, test_docs, selected_suffixes, unique_vocab)
 
        ### TRAIN AND EVALUATE ###
        text_clf = DecisionTreeClassifier(random_state=42)
        text_clf.fit(train_data, train_targets)
        predicted = text_clf.predict(test_data)

        accuracy = np.mean(predicted == test_targets)
        self._my_print(accuracy)
        self._my_print(metrics.confusion_matrix(test_targets, predicted))

        # Store in the user classifier dict
        dt_obj = DecisionTreeObj(text_clf, selected_suffixes, unique_vocab)
        self._clf_dict[user_id] = dt_obj

        return TrainingResult(accuracy, len(train_set))
    
    def _my_print(self, text: str) -> None:
        if (self._display):
            print(text)

    ### INITIALIZE DATASET ###
    def _initialize_dataset(self, source, source_docs, selected_suffixes, unique_vocab):
        all_features = []
        targets = []

        for (sent, label) in source:
            feature_list=[]

            # Average number of characters per word (length: 1)
            feature_list.append(EngineDT._avg_number_chars(sent))
            
            # Number of words per sentence (length: 1)
            feature_list.append(EngineDT._number_words(sent))

            # Counts for each stop word (length: number of stop words ~326)
            counts = EngineDT.word_counts(sent)
            for word in STOP_WORDS:
                if word in counts.keys():
                    feature_list.append(counts.get(word))
                else:
                    feature_list.append(0)
            
            # Proportion of words per sentence which are stop_words (length: 1)
            feature_list.append(EngineDT.proportion_words(sent, STOP_WORDS))

            # Proportion of POS counts (length: 14)
            p_counts = EngineDT.pos_counts(sent, source_docs, pos_list)
            for pos in p_counts.keys():
                feature_list.append(float(p_counts.get(pos))/float(len(sent)))
            
            # Proportion of suffix counts (length: variable)
            s_counts = EngineDT.suffix_counts(sent, source_docs, selected_suffixes)
            for suffix in s_counts.keys():
                feature_list.append(float(s_counts.get(suffix))/float(len(sent)))
            
            # Unique word count
            u_counts = EngineDT.unique_counts(sent, unique_vocab)
            for word in u_counts.keys():
                feature_list.append(u_counts.get(word))
            
            all_features.append(feature_list)
            targets.append(label) # need to parse as int? or 1/0?
        return all_features, targets


    ###### FEATURE EXTRACTORS ######
    @staticmethod
    def _avg_number_chars(text):
        total_chars = 0.0
        for word in text:
            total_chars += len(word)
        return float(total_chars)/float(len(text))
 
    @staticmethod
    def _number_words(text):
        return float(len(text))

    @staticmethod
    def word_counts(text):
        counts = {}
        for word in text:
            counts[word.lower()] = counts.get(word.lower(), 0) + 1
        return counts

    @staticmethod
    def proportion_words(text, wordlist):
        count = 0
        for word in text:
            if word.lower() in wordlist:
                count += 1
        return float(count)/float(len(text))
    
    @staticmethod
    def preprocess(source):
        source_docs = {}
        for (sent, _) in source:
            text = " ".join(sent)
            source_docs[text] = nlp(text)
        print("Dataset processed")
        return source_docs

    @staticmethod
    def pos_counts(text, source_docs, pos_list):
        pos_counts = {}
        doc = source_docs.get(" ".join(text))
        tags = []
        for word in doc:
            tags.append(str(word.tag_)[0])
        counts = Counter(tags)
        for pos in pos_list:
            if pos in counts.keys():
                pos_counts[pos] = counts.get(pos)
            else: pos_counts[pos] = 0
        return pos_counts

    @staticmethod
    def select_suffixes(train_docs, cutoff):
        all_suffixes = []
        for doc in train_docs.values():
            for word in doc:
                all_suffixes.append(str(word.suffix_).lower())
        counts = Counter(all_suffixes)
        sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        selected_suffixes = []
        for i in range(0, round(len(counts)*cutoff)):
            selected_suffixes.append(sorted_counts[i][0])
        return selected_suffixes
    
    @staticmethod
    def suffix_counts(text, source_docs, suffix_list):
        suffix_counts = {}
        doc = source_docs.get(" ".join(text))
        suffixes = []
        for word in doc:
            suffixes.append(str(word.suffix_))
        counts = Counter(suffixes)
        for suffix in suffix_list:
            if suffix in counts.keys():
                suffix_counts[suffix] = counts.get(suffix)
            else: suffix_counts[suffix] = 0
        return suffix_counts

    @staticmethod
    def unique_vocabulary(train_set, label1, label2, cutoff):
        voc1 = []
        voc2 = []
        for (sent, label) in train_set:
            if label==label1:
                for word in sent:
                    voc1.append(word.lower())
            elif label==label2:
                for word in sent:
                    voc2.append(word.lower())
        counts1 = Counter(voc1)
        sorted_counts1 = sorted(counts1.items(), key=operator.itemgetter(1), reverse=True)
        counts2 = Counter(voc2)
        sorted_counts2 = sorted(counts2.items(), key=operator.itemgetter(1), reverse=True)
        unique_voc = []
        for i in range(0, round(len(sorted_counts1)*cutoff)):
            if not sorted_counts1[i][0] in counts2.keys():
                unique_voc.append(sorted_counts1[i][0])
        for i in range(0, round(len(sorted_counts2)*cutoff)):
            if not sorted_counts2[i][0] in counts1.keys():
                unique_voc.append(sorted_counts2[i][0])
        return unique_voc

    @staticmethod
    def unique_counts(text, unique_voc):
        unique_counts = {}
        words = []
        for word in text:
            words.append(word.lower())
        counts = Counter(words)
        for word in unique_voc:
            if word in counts.keys():
                unique_counts[word] = counts.get(word)
            else: unique_counts[word] = 0
        return unique_counts