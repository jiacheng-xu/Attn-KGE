import cPickle as pkl

import numpy as np

file_name = '../data/fb_100_50.pkl'
new_filename = '../data/fb_100_100.pkl'


with open(file_name, 'r') as f:
    x = pkl.load(f)
    texts = x['texts']  # 40943 list
    ent_emb = x['entity_emb']  # np 40943,20
    ent_num = x['entity_num']
    word_emb = x['word_emb']
    print(ent_emb.shape)
    print(word_emb.shape)



print 'Start Prepoccessing'
tmp_word = np.zeros((word_emb.shape[0], ent_emb.shape[1]),dtype=float)
count = np.zeros((word_emb.shape[0],),dtype=float)
print tmp_word.shape
print count.shape

for i in xrange(ent_num):
    txt = texts[i]
    for idx in xrange(len(txt)):
        word_id = txt[idx]

        old_num = count[word_id]
        count[word_id] += 1
        new_num = count[word_id]
        tmp_word[word_id] = (old_num/new_num) * tmp_word[word_id] + (1./new_num) * ent_emb[i]

    if (i%1000==0):
        print i

x['word_emb'] = tmp_word
with open(new_filename, 'wb') as file:
    pkl.dump(x,file)

# check
with open(new_filename, 'r') as f:
    x = pkl.load(f)
    # texts = x['texts']  # 40943 list
    # ent_emb = x['entity_emb']  # np 40943,20
    # ent_num = x['entity_num']
    word_emb = x['word_emb']
    print(word_emb.shape)