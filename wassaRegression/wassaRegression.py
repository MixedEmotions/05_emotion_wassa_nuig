
# coding: utf-8

# In[2]:

from __future__ import division
import logging
import os
import xml.etree.ElementTree as ET

from senpy.plugins import EmotionPlugin, SenpyPlugin
from senpy.models import Results, EmotionSet, Entry, Emotion

logger = logging.getLogger(__name__)

# added packages
import codecs, csv, re, nltk
import numpy as np
import math, itertools
from drevicko.twitter_regexes import cleanString, setupRegexes, tweetPreprocessor
import preprocess_twitter
from collections import defaultdict
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.svm import LinearSVR, SVR

from nltk.tokenize import TweetTokenizer
import nltk.tokenize.casual as casual

import gzip
from datetime import datetime 

os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import load_model, model_from_json

class wassaRegression(EmotionPlugin):
    
    def __init__(self, info, *args, **kwargs):
        super(wassaRegression, self).__init__(info, *args, **kwargs)
        self.name = info['name']
        self.id = info['module']
        self._info = info
        local_path = os.path.dirname(os.path.abspath(__file__)) 
        
        self._maxlen = 55   
        self.WORD_FREQUENCY_TRESHOLD = 2
        
        self._savedModelPath = local_path + "/classifiers/LSTM/wassaRegression"
        self._path_wordembeddings = os.path.dirname(local_path) + '/glove.twitter.27B.100d.txt.gz'
        
        self._paths_ngramizer = local_path + '/wassa_ngramizer.dump'
        self._paths_svr = local_path + "/classifiers/SVR"
        self._paths_linearsvr = local_path + "/classifiers/LinearSVR"
        self.extension_classifier = '.dump'
        self._paths_word_freq = os.path.join(os.path.dirname(__file__), 'wordFrequencies.dump')
        
#         self._emoNames = ['sadness', 'disgust', 'surprise', 'anger', 'fear', 'joy'] 
        self._emoNames = ['anger','fear','joy','sadness']      
        
        

    def activate(self, *args, **kwargs):
        
        np.random.seed(1337)
        
        st = datetime.now()
        self._wordFrequencies = self._load_unique_tokens(filename = self._paths_word_freq)
        logger.info("{} {}".format(datetime.now() - st, "loaded _wordFrequencies"))
        
        self._wassaRegressionDLModels = {emo:self._load_model_emo_and_weights(self._savedModelPath, emo) for emo in self._emoNames}  
        
        st = datetime.now()
        self._Dictionary, self._Indices = self._load_original_vectors(
            filename = self._path_wordembeddings, 
            sep = ' ',
            wordFrequencies = self._wordFrequencies, 
            zipped = True) # leave wordFrequencies=None for loading the entire WE file
        logger.info("{} {}".format(datetime.now() - st, "loaded _wordEmbeddings"))
        
#         load_svr = False
        load_svr = True
        if load_svr:
            
            st = datetime.now()
            self._stop_words = get_stop_words('en')
            logger.info("{} {}".format(datetime.now() - st, "loaded _stop_words"))
            
            st = datetime.now()
            self._ngramizer = joblib.load(self._paths_ngramizer)
            logger.info("{} {}".format(datetime.now() - st, "loaded _ngramizers"))
            
            self._wassaRegressionSVMmodels = {
                'LinearSVR': self._load_classifier(PATH=self._paths_linearsvr, ESTIMATOR='LinearSVR' ),
                'SVR': self._load_classifier(PATH=self._paths_svr, ESTIMATOR='SVR')
                }

        logger.info("wassaRegression plugin is ready to go!")
        
    def deactivate(self, *args, **kwargs):
        try:
            logger.info("wassaRegression plugin is being deactivated...")
        except Exception:
            print("Exception in logger while reporting deactivation of wassaRegression")

            
            
    # CUSTOM FUNCTIONS
    
    def _text_preprocessor(self, text):
        
        text = preprocess_twitter.tokenize(text)
        
        text = casual.reduce_lengthening(text)
        text = cleanString(setupRegexes('twitterProAna'),text)  
        text = ' '.join([span for notentity,span in tweetPreprocessor(text, ("urls", "users", "lists")) if notentity]) 
        text = text.replace('\t','')
        text = text.replace('< ','<').replace(' >','>')
        text = text.replace('):', '<sadface>').replace('(:', '<smile>')
        text = text.replace(" 't", "t").replace("#", "")
        return ' '.join(text.split())

    def tokenise_tweet(text):
        text = preprocess_twitter.tokenize(text)
        text = preprocess_tweet(text)     
        return ' '.join(text.split()) 
    
    def _load_original_vectors(self, filename = 'glove.27B.100d.txt', sep = ' ', wordFrequencies = None, zipped = False):
       
        def __read_file(f):
            Dictionary, Indices  = {},{}
            i = 1
            for line in f:
                line_d = line.decode('utf-8').split(sep)

                token = line_d[0]
                token_vector = np.array(line_d[1:], dtype = 'float32')   
                if(wordFrequencies):
                    if(token in wordFrequencies):                
                        Dictionary[token] = token_vector
                        Indices.update({token:i})
                        i+=1
                else:
                    Dictionary[token] = token_vector
                    Indices.update({token:i})
                    i+=1
            return(Dictionary, Indices)
            
        if zipped:
            with gzip.open(filename, 'rb') as f:
                return(__read_file(f))
        else:
            with open(filename, 'rb') as f:
                return(__read_file(f))    
    
    
    
    # ===== SVR
    
    def _tweetToNgramVector(self, text):        
        return self._ngramizer.transform([text,text]).toarray()[0]       

    def _tweetToWordVectors(self, tweet, fixedLength=False):
        output = []    
        if fixedLength:
            for i in range(100):
                output.append(blankVector)
            for i,token in enumerate(tweet.split()):
                if token in self._Dictionary:
                    output[i] = self._Dictionary[token]                
        else:
             for i,token in enumerate(tweet.lower().split()):
                if token in self._Dictionary:
                    output.append(self._Dictionary[token])            
        return output
    
    
    def _ModWordVectors(self, x, mod=True):
        if len(x) == 0:       
            if mod:
                return np.zeros(self.EMBEDDINGS_DIM*3, dtype='float32')
            else:
                return np.zeros(self.EMBEDDINGS_DIM, dtype='float32')
        m = np.matrix(x)
        if mod:
            xMean = np.array(m.mean(0))[0]
            xMin = np.array(m.min(0))[0]
            xMax = np.array(m.max(0))[0]
            xX = np.concatenate((xMean,xMin,xMax))
            return xX
        else:
            return np.array(m.mean(0))[0]
        
    def _bindTwoVectors(self, x0, x1):
        return np.array(list(itertools.chain(x0,x1)),dtype='float32') 
    
    def _bind_vectors(self, x):
        return np.concatenate(x)  
    
    def _load_classifier(self, PATH, ESTIMATOR):
        
        models = []
        st = datetime.now()

        for EMOTION in self._emoNames:
            filename = os.path.join(PATH, EMOTION + self.extension_classifier)
            st = datetime.now()
            m = joblib.load(filename)
            logger.info("{} loaded _wassaRegression.{}.{}".format(datetime.now() - st, ESTIMATOR, EMOTION))
            models.append( m )
            
        return models
    
    def _load_unique_tokens(self, filename = 'wordFrequencies.dump'):    
        return joblib.load(filename)
        
    def _convert_text_to_vector(self, text, text_input):
        
        ngramVector = self._tweetToNgramVector(text)
        embeddingsVector = self._ModWordVectors(self._tweetToWordVectors(text))
        
        X = np.asarray( self._bind_vectors((ngramVector, embeddingsVector)) ).reshape(1,-1)   
        return X

    
    
    
    # ===== LSTM
    
    def _load_model_emo_and_weights(self, filename, emo):
        st = datetime.now()
        with open(filename+'.'+emo+'.json', 'r') as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            
        loaded_model.load_weights(filename+'.'+emo+'.h5')
        logger.info("{} {}".format(datetime.now() - st, "loaded _wassaRegression.LSTM."+emo))
        return loaded_model
    
    def _lists_to_vectors(self, text):
        train_sequences = [self._text_to_sequence(text)]  
        X = sequence.pad_sequences(train_sequences, maxlen=self._maxlen)

        return X
    
    def _text_to_sequence(self,text):
        train_sequence = []
        for token in text.split():
            try:
                train_sequence.append(self._Indices[token])
            except:
                train_sequence.append(0)
        train_sequence.extend([0]*( self._maxlen-len(train_sequence)) )
        return np.array(train_sequence)                  

    def _extract_features(self, X):  
        feature_set = {}
        for emo in self._emoNames:
            feature_set.update({emo:self._wassaRegressionDLModels[emo].predict(X)[0][0]})
            
        return feature_set 
    
    def _extract_features_svr(self, X):  
        if self.ESTIMATOR in ['SVR','LinearSVR']:
            feature_set = {
                emo: float(clf.predict(X)[0]) for emo,clf in zip(self._emoNames, self._wassaRegressionSVMmodels[self.ESTIMATOR])}
        else:
            feature_set = {
                emo: float(clf.predict(X)[0]) for emo,clf in zip(self._emoNames, self._wassaRegressionSVMmodels['LinearSVR'])}
        return feature_set 
    
    
    # ANALYZE    
    
    def analyse(self, **params):
        logger.debug("wassaRegression LSTM Analysing with params {}".format(params))          
        
        st = datetime.now()
           
        text_input = params.get("input", None)        
        text = self._text_preprocessor(text_input) 
        
        self.ESTIMATOR = params.get("estimator", 'LSTM')
        
        if self.ESTIMATOR == 'LSTM':        
            X_lstm = self._lists_to_vectors(text = text)           
            feature_text = self._extract_features(X_lstm) 
            
        elif self.ESTIMATOR == 'averaged':
            X_lstm = self._lists_to_vectors(text = text)
            X_svr = self._convert_text_to_vector(text=text, text_input=text_input) 
            
            feature_text_lstm = self._extract_features(X_lstm)
            feature_text_svr = self._extract_features_svr(X_svr) 
            
            feature_text = {emo:np.mean([feature_text_lstm[emo], feature_text_svr[emo]]) for emo in self._emoNames}
            
        else:     
            X_svr = self._convert_text_to_vector(text=text, text_input=text_input)            
            feature_text = self._extract_features_svr(X_svr)  
        
        logger.info("{} {}".format(datetime.now() - st, "string analysed"))
            
        response = Results()
       
        entry = Entry()
        entry.nif__isString = text_input
        
        emotionSet = EmotionSet()
        emotionSet.id = "Emotions"
        
        emotionSet.onyx__maxIntensityValue = float(100.0)
        
        emotion1 = Emotion() 
        for dimension in ['V','A','D']:
            weights = [feature_text[i] for i in feature_text]
            if not all(v == 0 for v in weights):
                value = np.average([self.centroids[i][dimension] for i in feature_text], weights=weights) 
            else:
                value = 5.0
            emotion1[self.centroid_mappings[dimension]] = value      

        emotionSet.onyx__hasEmotion.append(emotion1)    
        
        for i in feature_text:
            emotionSet.onyx__hasEmotion.append(Emotion(onyx__hasEmotionCategory=self.wnaffect_mappings[i],
                                    onyx__hasEmotionIntensity=float(feature_text[i])*emotionSet.onyx__maxIntensityValue))
        
        entry.emotions = [emotionSet,]
        
        response.entries.append(entry)  
            
        return response

