from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, multiply
from keras.regularizers    import l2
from keras.layers.core     import Reshape
from keras.layers.normalization import BatchNormalization as BN
import keras.backend as K
import numpy as np
import random
import sys
import pickle
import glob
import copy
import os
import re
import gzip
import json

VOC_SIZE = 3504
in_timesteps   = 300
out_timesteps  = 100
inputs      = Input(shape=(in_timesteps, VOC_SIZE))
encoded     = GRU(1024)(inputs)
encoder     = Model(inputs, encoded)

x           = RepeatVector(out_timesteps)(encoded)
x           = Bi(GRU(1500, return_sequences=True))(x)
#x           = TD(Dense(VOC_SIZE*2, activation='relu'))(x)
decoded     = TD(Dense(VOC_SIZE, activation='softmax'))(x)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer=Adam(), loss='categorical_crossentropy')

buff = None
def callbacks(epoch, logs):
  global buff
  buff = copy.copy(logs)
  print("epoch" ,epoch)
  print("logs", logs)

def train():
  if '--resume' in sys.argv:
    model = sorted( glob.glob("models/*.h5") ).pop()
    print("loaded model is ", model)
    autoencoder.load_weights(model)
  
  print_callback = LambdaCallback(on_epoch_end=callbacks)

  counter = 0
  for i in range(2000):
    for name in glob.glob('dataset/*.pkl'):
      pairs = pickle.loads(gzip.decompress(open(name,'rb').read()))
      idenses, cdenses = [], []
      for idense, cdense in pairs:
        idenses.append(idense)
        cdenses.append(cdense)
      Xs, Ys = np.array(idenses), np.array(cdenses)
      batch_size = random.randint( 3, 32 )
      random_optim = random.choice( [Adam(), SGD(), RMSprop()] )
      autoencoder.fit(Xs, Ys, 
          shuffle=True, 
          batch_size=batch_size, epochs=random.randint(1,5), 
          callbacks=[print_callback] )
      if counter%10 == 0:
        autoencoder.save("models/{:09d}_{:09f}.h5".format(counter, buff['loss']))
      counter += 1

def predict():
  for name in glob.glob('dataset/000000249.pkl'):
    pairs = pickle.loads(gzip.decompress(open(name,'rb').read()))
    idenses, cdenses = [], []
    for idense, cdense in pairs:
      idenses.append(idense)
      cdenses.append(cdense)
  Xs, Ys = np.array(idenses), np.array(cdenses)
  
  model = sorted( glob.glob("models/*.h5") ).pop()
  print("loaded model is ", model)
  autoencoder.load_weights(model)
  
  term_index = json.loads( open('intro_term_index.json').read() )
  term_index['<EOS>'] = len(term_index)
  term_index['<UNK>'] = len(term_index)
  index_term = { index:term for term, index in term_index.items() }
  Ys = autoencoder.predict( Xs )
  for ys, xs in zip(Ys.tolist(), Xs.tolist()):
    print('INPUT')
    for x in xs:
      xi = np.argmax(x)
      if index_term[xi] == '<EOS>':
        print('<EOS>')
        break
      print( index_term[xi], end='' )
    print('OUTOUT')
    for y in ys:
      yi = np.argmax(y)
      if index_term[yi] == '<EOS>':
        print('<EOS>')
        break
      print( index_term[yi], end='' )
    print()

if __name__ == '__main__':
  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
