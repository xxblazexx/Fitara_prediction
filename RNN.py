import pandas as pd
import spacy
import nltk
from spacy.lang.en import English
import re
from spacy.lang.en.stop_words import STOP_WORDS
import tensorflow as tf
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import datetime
from nltk.stem.porter import PorterStemmer 
import pickle
from tensorflow import keras
#%reload_ext tensorboard

nlp=spacy.load("en_core_web_sm")
data=pd.read_csv("extract_combined.csv")
label=pd.read_csv("labels.csv")
df=pd.DataFrame(data)
df.head()

''' Removal of \characters '''

def text_processing(df):
  
  ''' Remove \n and other backslash characters and convert to lower case '''
  
  df['processed']=df['text'].apply(lambda x: re.sub(r'[^\w+ ]',' ',x.lower()))
  
  ''' Remove Digits '''
  
  df['processed'] = df['processed'].apply(lambda x: re.sub(r'[0-9]',' ',x))
  
  #print((df.loc[1]['processed']))
  
  ''' Remove special characters that aren't required '''
  
  df['processed'] = df['processed'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;_]',' ',x))

  #df['processed'].head()
  #print((df.loc[0]['processed']))


  #print(len(df.loc[0]['processed']))
  
  ''' Remove stop word '''
 
  df['processed'] = df['processed'].apply(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))
  
  '''Stemming of text'''
  
  ps=PorterStemmer()
  df['processed'] = df['processed'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
  #df['processed'].head() 
  #length after removal of stop words
  #print(len(df.loc[0]['processed']))

  ''' Counting number of words '''

  df['words'] = df['processed'].apply(lambda x: len(str(x).split(' ')))
  
  #df['words1'] = df['text'].apply(lambda x: len(str(x).split(' ')))
  #print(df['words'].head())
  #' '.join([nlp(word) for word in x.split()]
  return df

text_processing(df);
''' MERGING labels with extracted_combined '''

newdata=pd.merge(data, label[['document_name','is_fitara']], on='document_name')

#newdata.head()
#data=newdata

df1=pd.DataFrame(newdata)
df1.head()

''' Change label to 1 or 0 from yes or no '''

df1['is_fitara']=df1['is_fitara'].apply(lambda x:0 if(x == 'No') else 1 )

print(df1.head())
''' The maximum number of words to be used. (most frequent) '''

MAX_NB_WORDS = 50000

''' Max number of words in each complaint. '''

MAX_SEQUENCE_LENGTH = 250

''' This is fixed. '''

EMBEDDING_DIM = 100


def tokenize(df):
  ''' Tokenizing the processed data '''
  
  tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
  tokenizer.fit_on_texts(df['processed'].values)
  word_index = tokenizer.word_index
  
  print(len(word_index))
  
  ''' setting up the appropriate shape by padding according to model '''

  X = tokenizer.texts_to_sequences(df['processed'].values)
  X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
  
  #Y=pd.get_dummies(df1['is_fitara']).values

  #print('Shape of label tensor:', Y.shape)
  print('Shape of data tensor:', X.shape)
  return X

X=tokenize(df);
#pickle.dump(tokenize(df))
#from sklearn.model_selection import train_test_split

''' Creating Train and Test data '''

X_train, X_test, Y_train, Y_test = train_test_split(X,df1['is_fitara'], test_size = 0.10, random_state = 42)

#print(type(X_train))
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
#print(Y_train)
''' Over Sampling of data '''

sm = SMOTE(random_state=2)
X_train_res, Y_train_res = sm.fit_sample(X_train, Y_train)

print(X_train_res.shape, Y_train_res.shape)
print(X_test.shape, Y_test.shape)
''' Populating dummy values to target data to get proper shape of array '''

Y_train=pd.get_dummies(Y_train_res).values

#Y_train.shape

Y_test=pd.get_dummies(Y_test).values

#Y_test.shape

''' Building the LSTM Model '''
def lmodel(X,X_train_res,Y_train):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
  model.add(tf.keras.layers.SpatialDropout1D(0.2))
  model.add(tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
  #model.add(tf.keras.layers.BatchNormalization(0.7))
  model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  epochs = 5
  batch_size = 64  
  history = model.fit(X_train_res, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
  return model
epochs = 5
batch_size = 64

'''Training the LSTM Model with train data and validation data'''
#print(X_test,Y_test)

#history = model.fit(X_train_res, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
plswork=lmodel(X,X_train_res,Y_train)
#pickle.dumps(history)

'''Saving model so that it can be accessed in another process'''

#pickle.dump(plswork,open('work','wb'))
plswork.save('C:/Users/vchidambaram/Desktop/mymodel.h5')
