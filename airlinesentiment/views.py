from django.shortcuts import render
from django.http import HttpResponse
import os
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer
nltk.download('punkt')
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
import time
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
from tensorflow.python.keras.models import load_model



modulePath = os.path.dirname(__file__)  # get current directory
filePath = os.path.join(modulePath, 'tokenizer.pickle')
with open(filePath, 'rb') as f:
	tokenizer=pickle.load(f)
    
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

filePath = os.path.join(modulePath, 'sentiment.h5')
model = load_model(filePath)
model._make_predict_function()


def datacleaning(text): 

          text=re.sub(r'@\w+', '', text)
          text=re.sub(r"https?://\S+|www\.\S+",'',text)

          text=re.sub("[^a-zA-Z]",' ',text)
          text=text.lower()
          text=text.split()
          stopword = stopwords.words('english')
          text= [word for word in text if (word not in stopword )]
          text=' '.join(text)
          stem=[]
          stopword = stopwords.words('english')
          snowball_stemmer = SnowballStemmer('english')
          word_tokens = nltk.word_tokenize(text)
          stemmed_word = [snowball_stemmer.stem(word) for word in word_tokens]
          stem=' '.join(stemmed_word)

          return stem
      
        

#def predict
      
        

def home(request):
    return render(request, 'index.html')


def classify(request):
    #Get the text
    global graph
    global sess
    max_length=80
    trunc_type="post"
    oov_tok="<OOV>"
    padding_type="post"
    djtext = request.GET.get('text', 'default')
    if djtext != "default":
        text =datacleaning(djtext) 
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=max_length ,padding=padding_type, truncating=trunc_type)
        with graph.as_default():
        	set_session(sess)
        	predicted = model.predict(padded)

        print(predicted)
        if predicted[0] >=0.5:
            cat =  "Positive"
        else:
            cat = "Negative"
        
    	
    if djtext=="default":
    	predicted = "No Text Provided please try again"

    params = {'Category': cat}
    return render(request, 'result.html', params)


   
  

