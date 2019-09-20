from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import numpy as np
import pickle as p
import json
import pandas as pd
import spacy
import nltk
from spacy.lang.en import English
import re
from spacy.lang.en.stop_words import STOP_WORDS
import tensorflow as tf
from imblearn.over_sampling import SMOTE
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import datetime
from nltk.stem.porter import PorterStemmer 
import pickle
from tensorflow import keras
import sqlite3


app = Flask(__name__)
nlp=spacy.load("en_core_web_sm")

@app.route('/')
def hworld():
    return render_template('webpage.html')

@app.route('/list')
def list():

   #CONNECTING TO THE FITARA DATASET
   con = sqlite3.connect('mydb.db')
   
   #CREATE A ROW OBJECT
   con.row_factory = sqlite3.Row
   
   #CREATE A CURSOR
   cur = con.cursor()
   cur.execute("select * from result")
   
   #GET ALL THE ROWS
   rows = cur.fetchall();	
   #print(rows)
   #PASS THE ROW VALUES TO LIST.HTML
   return render_template('list.html',rows = rows)
    

@app.route('/preprocess', methods=['POST','GET'])
def preprocess():
    
    '''RNN Model'''
    
    if request.method=='POST':
        data = request.form['Input data']
        #print(data)
        #x=data
        odata=data
        
        '''connecting to the sqlite3 database'''
        
        con=sqlite3.connect('mydb.db')
        cursorObj=con.cursor()
        #cursorObj.execute("CREATE TABLE result(input_data text,result text)")
        con.commit()
        '''preprocess the data'''
        
        data1 =(lambda x: re.sub(r'[^\w+ ]',' ',x.lower()))
        data=data1(data)
        #print(data1)
        data2 =(lambda x: re.sub(r'[0-9]',' ',x))
        data=data2(data)
        data3 =(lambda x: re.sub(r'[/(){}\[\]\|@,;_]',' ',x))
        data=data3(data)
        data4 =(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))
        data=data4(data)
        ps=PorterStemmer()
        data5 =(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
        data=data5(data)
        
        #print(data)
        #data = request.get_json()
        
        '''tokenize the pre procesed data'''
        MAX_NB_WORDS = 50000
        MAX_SEQUENCE_LENGTH = 250
        EMBEDDING_DIM = 100
        
        #data = request.form['inputdata']
        
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts([data])
        
        #word_index = tokenizer.word_index
        
        X1 = tokenizer.texts_to_sequences([data])
        X1 = tf.keras.preprocessing.sequence.pad_sequences(X1, maxlen=MAX_SEQUENCE_LENGTH)
        prediction = np.array2string(modelfile.predict(X1))
        en=(odata,prediction)
        '''inserting input data and result into sqlite3 database'''
        
        cursorObj.execute("INSERT INTO result(input_data,result) VALUES(?,?)",en)
        con.commit()
        con.close()
        
        #return jsonify(prediction)
        #df['words'] = df['processed'].apply(lambda x: len(str(x).split(' ')))
        
        return render_template('hope.html',data=prediction)
    
    else:
        '''getting data as input from user'''
        
        return render_template('webpage.html')


@app.route('/bayesmodel', methods=['POST','GET'])
def bayesmodel():
    
    '''Naive Bayes model is implemented here by preprocessing and count vecotrizing the sent data'''
    
    if request.method=='POST':
        data = request.form['Input data']
        
        #print(data)
        #x=data
        #odata=data
        #cursorObj.execute("CREATE TABLE result(input_data text,result text)")
        #con.commit()
        
        '''pre preocessing of data'''
        
        data1 =(lambda x: re.sub(r'[^\w+ ]',' ',x.lower()))
        data=data1(data)
        
        #print(data1)
        
        data2 =(lambda x: re.sub(r'[0-9]',' ',x))
        data=data2(data)
        data3 =(lambda x: re.sub(r'[/(){}\[\]\|@,;_]',' ',x))
        data=data3(data)
        data4 =(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))
        data=data4(data)
        ps=PorterStemmer()
        data5 =(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
        data=data5(data)
        '''loading the pipelinecount vectorizer having the vocabulary from naivebayes.py'''
        cv=p.load(open('C:/Users/vchidambaram/Desktop/cvpipeline.pickle','rb'))
        data=cv.transform([data])
        
        #from sklearn.externals import joblib
        #cv=joblib.load(pipeline.pickle)
        #data=cv.fit_transform([data])
        #print(data.shape)
        
        prediction = np.array2string(model.predict(data))
        return jsonify(prediction)
    
    else:
        '''Getting data as input from user'''
        
        return render_template('naivewebpage.html')
    

'''
cant convert data from string back to int properly at client side

@app.route('/tokenizer/', methods=['POST'])
def toke():
    MAX_NB_WORDS = 50000
    MAX_SEQUENCE_LENGTH = 250
    EMBEDDING_DIM = 100
    data = request.get_json()
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(data)
    #word_index = tokenizer.word_index
    X1 = tokenizer.texts_to_sequences(data)
    X1 = tf.keras.preprocessing.sequence.pad_sequences(X1, maxlen=MAX_SEQUENCE_LENGTH)
    X1=np.array2string(X1)
    return jsonify(X1)
'''

'''
@app.route('/api', methods=['POST'])
def makecalc():
    #data = request.get_json()
    MAX_NB_WORDS = 50000
    MAX_SEQUENCE_LENGTH = 250
    EMBEDDING_DIM = 100
    data = request.form['inputdata']
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts([data])
    #word_index = tokenizer.word_index
    X1 = tokenizer.texts_to_sequences([data])
    X1 = tf.keras.preprocessing.sequence.pad_sequences(X1, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = np.array2string(modelfile.predict(X1))
    
    return jsonify(prediction)
'''

if __name__ == '__main__':
    modelfile = keras.models.load_model('C:/Users/vchidambaram/Desktop/mymodel.h5')
    naivemodelfile='C:/Users/vchidambaram/Desktop/naivemodel.pickle'
    model = p.load(open(naivemodelfile, 'rb'))
    app.run()
    
