import os, numpy, theano, time
import cPickle as pkl
from util import *
import getopt
import sys
# from lstm_launch import *
# from lstm_att_launch import *

from theano import config


def test(
):
    attention_path = 'att_new.pkl'
    import cPickle as pkl
    e2i, i2e, w2i, i2w = None, None, None, None
    with open('../data/e2i.pkl', 'r') as e:
        x = pkl.load(e)
        e2i = x['e2i']
        i2e = x['i2e']
        print('Len of %s %s' % ('e2i', len(e2i)))
    with open('../data/w2i.pkl', 'r') as e:
        x = pkl.load(e)
        w2i = x['w2i']
        i2w = x['i2w']
        print('Len of %s %s' % ('w2i', len(w2i)))

    with open('../data/r2i.pkl', 'r') as e:
        x = pkl.load(e)
        r2i = x['r2i']
        i2r = x['i2r']
        print('Len of %s %s' % ('r2i', len(r2i)))



    tparams = None
    with open('../data/' + attention_path, 'r') as main_file:
        x = pkl.load(main_file)
        __, tparams, options = x

    from lstm_launch import build_test
    options['model_name'] = 'lstm'
    test_text, test_mask, rel, test_end_step = build_test(tparams, options, tparams)
    f_test = theano.function(inputs=[test_text, test_mask,rel], outputs=[test_end_step],on_unused_input='ignore')

    rel_emb = tparams['rel_emb'].get_value()
    set =options['train_samples']
    for i in xrange(len(set)):
        h,t,r = set[i]
        print set[i]
        print h, t, r
        h_text=options['texts'][h]
        t_text=options['texts'][t]
        com=[h_text,t_text]
        input_list = [com]


        x, x_mask = prepare_data(input_list,
                             maxlen=None)
        # print x
        # print x_mask
        # print x.shape

        relation = rel_emb[r]
        # print relation
        end_step = f_test(x,x_mask,relation)
        print end_step

        print len(end_step)
        print end_step.shape
        break

    ######
    # test for gene test


test()
