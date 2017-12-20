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

in_timesteps   = 300
out_timesteps  = 100
inputs      = Input(shape=(in_timesteps, 2504))
encoded     = LSTM(512)(inputs)
encoder     = Model(inputs, encoded)

x           = RepeatVector(out_timesteps)(encoded)
x           = Bi(LSTM(512, return_sequences=True))(x)
decoded     = TD(Dense(2504, activation='softmax'))(x)

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
      batch_size = random.randint( 32, 64 )
      random_optim = random.choice( [Adam(), SGD(), RMSprop()] )
      autoencoder.fit(Xs, Ys, 
          shuffle=True, 
          batch_size=batch_size, epochs=1, 
          callbacks=[print_callback] )
      counter += 1
      if counter%10 == 0:
        autoencoder.save("models/{:09d}_{:09f}.h5".format(counter, buff['loss']))

def predict():
  c_i = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  i_c = { i:c for c, i in c_i.items() }
  xss = []
  heads = []
  with open("dataset/corpus.distinct.txt", "r") as f:
    for fi, line in enumerate(f):
      print("now iter ", fi)
      line = line.strip()
      head, tail = line.split("___SP___")
      heads.append( head ) 
      xs = [ [0.]*128 for _ in range(50) ]
      for i, c in enumerate(head): 
        xs[i][c_i[c]] = 1.
      xss.append( np.array( list(reversed(xs)) ) )
    
  Xs = np.array( xss )
  print( Xs)
  model = sorted( glob.glob("models/*.h5") ).pop(0)
  print("loaded model is ", model)
  autoencoder.load_weights(model)

  Ys = autoencoder.predict( Xs ).tolist()
  for ez, (head, y) in enumerate(zip(heads, Ys)):
    terms = []
    for v in y:
      term = max( [(s, i_c[i]) for i,s in enumerate(v)] , key=lambda x:x[0])[1]
      terms.append( term )
    tail = re.sub(r"」.*?$", "」", "".join( terms ) )
    print(ez, head, "___SP___", tail )
if __name__ == '__main__':
  if '--test' in sys.argv:
    test()

  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
