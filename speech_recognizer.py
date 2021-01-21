import os
import argparse
import warnings

import numpy as np
from scipy.io import wavfile

from hmmlearn import hmm #HMM stands for "Hidden Markov Models"
from python_speech_features import mfcc

#function for analyzing arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Trains / the HMM-based speech recognition system")
    parser.add_argument("--input_folder", dest="input_folder",
                       required=True, help="Input folder containig the audio files for training")
    return parser
    
#class for training HMM models
class ModelHmm(object):
    def __init__(self, num_components=4, num_iter=1000):
        self.n_components = num_components
        self.n_iter = num_iter
        
        #type of covariation and HMM
        self.cov_type = "diag"
        self.model_name = "GaussianHMM"
        
        #variable for storing each model
        self.models = []
        
        #Определим модель, используя указанные параметры
        self.model = hmm.GaussianHMM(n_components=self.n_components,
                                    covariance_type=self.cov_type,
                                    n_iter=self.n_iter)
        
    #Определим метод для обучения модели
    #training_data-это 2D массив numpy, в котором каждая строка имеет 13 измерений
    def train(self, training_data):
        np.seterr(all="ignore")
        cur_model = self.fit.model(training_data)
        self.models.append(cur_model)
        
    #Определим метод, оценивающий входные данные
    def compute_score(self, input_data):
        return self.model.score(input_data)
        
#function creating model for each word in training data
def build_models(input_folder):
    #var for storing all models
    speech_models = []
    
    #анализ входного канала
    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)
        
        if not os.path.isdir(subfolder):
            continue
        
        #Metka(na nahozhdenie slova)
        label = subfolder[subfolder.rfind('/')+1:]
        
        #variable created for storing training data
        X = np.array([])
        
        #creating list of files which will be used for training model, one file in each folder is saved for testing
        training_files = [x for x in os.path.listdir(subfolder) if x.endswith('.wav')][:-1]
        
        #building models by going through training_files
        for filename in training_files:
            #getting path to current file
            filepath = os.path.join(subfolder, filename)
            
            #read audio from current folder
            freq, audio = wavfile.read(filepath)
            
            #get MFCC features
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                features_mfcc = mfcc(freq, audio)
                
            #connect features_mfcc(point of data) to X
            if len(X) == 0:
                X = features_mfcc
            else:
                X = np.append(X, features_mfcc, axis=0)
                
        #Initialization of HMM
        #Create HMM
        model = ModelHMM()
        #Train HMM
        model.train(X)
        #Save model for current word
        speech_models.append((model, label))
        #Sbros peremennoi
        model = None
    return speech_models         

#function for testing training date
def run_tests(test_files):
    for test_file in test_files:
        freq, audio = wavfile.read(test_file)
        
        #get MFCC features
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            features_mfcc = mfcc(freq, audio)
                
                
        #var for maximum ocenka
        max_score = -float('inf')
        #var for output label
        output_label = None
        
        #iteration through HMM models and choosing the best one of them
        for item in speech_models:
            model, label = item
            
            #count score here and consider it with max_score
            score = model.compute_score(features_mfcc)
            if score>max_score:
                max_score = score
                predicted_label = label
            
        #vyvod predicted result
        start_index = test_file.find('/')+1
        end_index = test_file.rfind('/')
        original_label = test_file[start_index:end_index]
        print('Original: ', orignal_label)
        print('\nPredicted: ', predicted_label)              
        
#Main function
if __name__ == '__main__':
    #get input folder from input parameter
    args = build_arg_parser().parse_args()
    input_folder = args.input_folder
    
    #create HMM model for each word
    speech_models = build_models(input_folder)
    
    #1 file in each folder is used for testing. Let's use this file to find out how much precise 
    #is current model
    #--15 here is the 15'th file in each folder
    test_files = []
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if '10' in x):
            filepath = os.path.join(root, filename)
            test_files.append(filepath)
            
    run_tests(test_files)
            
    
    
    
    
    
    
    
    
