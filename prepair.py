import glob

import json

import MeCab

import sys

import random

import pickle

import gzip
if '--wakati' in sys.argv:
  dataset = []
  m = MeCab.Tagger('-Owakati')
  for name in glob.glob('../catchphrase_introduction/*.json'):
    try:
      obj = json.loads(open(name).read())
    except Exception as ex:
      continue
    print( obj )

    catchs = m.parse(obj['catch']).strip().split()
    intros = m.parse(obj['intro']).strip().split()
    dataset.append( (catchs, intros) )
  open('dataset.json','w').write( json.dumps(dataset, indent=2, ensure_ascii=False) )

if '--freq' in sys.argv:
  dataset = json.loads(open('dataset.json').read())

  term_freq = {}

  for catchs, intros in dataset:
    for term in intros:
      if term_freq.get(term) is None:
        term_freq[term] = 0
      term_freq[term] += 1

  terms = []
  for index, (term, freq) in enumerate(sorted(term_freq.items(), key=lambda x:x[1]*-1)):
    print(index, term, freq)
    terms.append(term)
    if index > 2500:
      break

  random.shuffle(terms)
  
  term_index = {}
  for index, term in enumerate(terms):
    term_index[term] = index

  open('intro_term_index.json', 'w').write( json.dumps(term_index, indent=2, ensure_ascii=False))

 
  term_index = {}
  for catchs, intros in dataset:
    for term in catchs: 
      if term_index.get(term) is None:
        term_index[term] = len(term_index)
  open('catch_term_index.json', 'w').write( json.dumps(term_index, indent=2, ensure_ascii=False))

if '--make' in sys.argv:
  dataset = json.loads(open('dataset.json').read())
  term_index = json.loads(open('intro_term_index.json').read())
  
  term_index['<EOS>'] = len(term_index)
  term_index['<UNK>'] = len(term_index)

  pairs = []
  for num, (catchs, intros) in enumerate(dataset):
    ibase = [term_index['<EOS>']]*300

    for index, term in enumerate(intros[:300]):
      if term_index.get(term) is None:
        ibase[index] = term_index['<UNK>']
      else:
        ibase[index] = term_index[term]
    
    idense = [ [0.0]*len(term_index) for i in range(300) ] 
    for index, b in enumerate(ibase):
      idense[index][b] = 1.0

    cbase = [term_index['<EOS>']]*100
    for index, term in enumerate(intros[:100]):
      if term_index.get(term) is None:
        cbase[index] = term_index['<UNK>']
      else:
        cbase[index] = term_index[term]
    cdense = [ [0.0]*len(term_index) for i in range(100) ]    
    for index, b in enumerate(cbase):
      cdense[index][b] = 1.0
    
    pairs.append( (idense, cdense) )
    if len(pairs) == 250:
      chunk = gzip.compress(pickle.dumps(pairs))
      open('dataset/{:09d}.pkl'.format(num), 'wb').write( chunk )
      pairs = []
