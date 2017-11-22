# -*-coding:utf-8-*-
__author__ = 'kanchan'
from util import *
from module import *
import theano


def transe(tparams, options, h_pos, t_pos, h_neg, t_neg, r_emb):
    if options['distance'] == 'l1':
        return tensor.maximum(0, options['margin'] + tensor.sum(tensor.abs_(h_pos + r_emb[:, 0, :] - t_pos), axis=1) - \
                              tensor.sum(tensor.abs_(h_neg + r_emb[:, 1, :] - t_neg), axis=1))
    elif options['distance'] == 'l2':
        return tensor.maximum(0, options['margin'] + tensor.sum(tensor.sqr(h_pos + r_emb[:, 0, :] - t_pos), axis=1) - \
                              tensor.sum(tensor.sqr(h_neg + r_emb[:, 1, :] - t_neg), axis=1))
    else:
        raise NotImplementedError('Illegal distance measure.')


def init_params(options):
    params = OrderedDict()

    if options['wdim'] != options['edim']:
        W = ortho_weight(indim=options['wdim'], outdim=options['edim'])
        params['cbow_W'] = W

    if options['model_name'].endswith('gate'):
        params['gate_emb'] = ortho_weight(options['ent_num'], options['edim'])
    elif options['model_name'].endswith('gate_pro'):
        params['gate_U'] = ortho_weight(options['edim'])
        params['gate_W'] = ortho_weight(options['edim'])
        params['gate_b'] = numpy.random.uniform(low=-.1, high=.1, size=(options['edim'],)).astype(config.floatX)
    elif options['model_name'].endswith('gate_s'):
        params['gate_s'] = numpy.random.uniform(low=-.1, high=.3, size=(options['ent_num'], 1)).astype(config.floatX)

    if options['model_name'].find('att') != -1:
        params = param_init_attention(options, params, prefix='attention', dim=options['edim'])

    params = param_init_cbow(options, params, prefix='cbow', in_dim=options['wdim'], out_dim=options['edim'])
    return params


def build_model(tparams, options, ttparams=None):
    if ttparams == None:
        ttparams = tparams

    print 'Model: cbow'

    use_noise = theano.shared(numpy_floatX(0.))

    # head and tail load
    ht = tensor.matrix('ht_triplet', dtype='int64')
    r = tensor.matrix('r_triplet', dtype='int64')

    n_samples = ht.shape[0]
    ent_emb = ttparams['ent_emb'][ht.flatten()]
    ent_emb = ent_emb.reshape([n_samples, 4, options['edim']])

    # relation load
    # Naive approach
    rel_emb = ttparams['rel_emb'][r.flatten()]
    rel_emb = rel_emb.reshape([n_samples, 2, options['edim']])

    text = tensor.matrix('text', dtype='int64')
    mask = tensor.matrix('text_mask')
    #
    # text input shape : seqth lenth, batch_size*4
    #
    # assert text.shape[1] == r.shape[0] * 4
    text_emb = tparams['word_emb'][text.flatten()]
    text_emb = text_emb.reshape([text.shape[0], text.shape[1], options['wdim']])

    # rt_text = lstm(tparams, text_emb, options, mask=mask,in_dim=options['wdim'],out_dim=options['edim'])
    # end_step = rt_text[-1]

    # rt_text = cbow(tparams, text_emb, options, prefix='cbow', mask=mask, in_dim=options['wdim'],
    #                out_dim=options['edim'])

    rt_text = text_emb
    if options['wdim'] != options['edim']:
        rt_text = theano.tensor.dot(text_emb, tparams['cbow_W'])

    if options['model_name'].find('att') != -1:
        h_pos_text, t_pos_text, h_neg_text, t_neg_text = rt_text[:, 0:r.shape[0], :], \
                                                         rt_text[:, r.shape[0]: r.shape[0] * 2, :], \
                                                         rt_text[:, r.shape[0] * 2:r.shape[0] * 3, :], \
                                                         rt_text[:, r.shape[0] * 3:r.shape[0] * 4, :]
        pos_relation = rel_emb[:, 0, :].reshape([n_samples, options['edim']])
        neg_relation = rel_emb[:, 1, :].reshape([n_samples, options['edim']])
        h_pos_text = attention(tparams, h_pos_text, pos_relation, options, prefix='attention',
                               mask=mask[:, 0:r.shape[0]])

        ####
        # f_att = options['f_att']
        # alp = f_att(h_pos_text, pos_relation,mask=mask[:, 0:r.shape[0]])
        # options['alp'] = alp

        ####

        t_pos_text = attention(tparams, t_pos_text, pos_relation, options, prefix='attention',
                               mask=mask[:, r.shape[0]:r.shape[0] * 2])
        h_neg_text = attention(tparams, h_neg_text, neg_relation, options, prefix='attention',
                               mask=mask[:, r.shape[0] * 2:r.shape[0] * 3])
        t_neg_text = attention(tparams, t_neg_text, neg_relation, options, prefix='attention',
                               mask=mask[:, r.shape[0] * 3:r.shape[0] * 4])
        end_step = concatenate([h_pos_text, t_pos_text, h_neg_text, t_neg_text], axis=0)  # 4 * nsamples, dim
    else:
        # h_pos_text, t_pos_text, h_neg_text, t_neg_text = h_pos_text[-1], t_pos_text[-1], h_neg_text[-1], t_neg_text[-1]
        # end_step = rt_text[-1]
        proj = (rt_text * mask[:, :, None]).sum(axis=0)
        end_step = proj / mask.sum(axis=0)[:, None]

        h_pos_text, t_pos_text, h_neg_text, t_neg_text = end_step[0:r.shape[0]], \
                                                         end_step[r.shape[0]: r.shape[0] * 2], \
                                                         end_step[r.shape[0] * 2:r.shape[0] * 3], \
                                                         end_step[r.shape[0] * 3:r.shape[0] * 4],

    # gate
    f_loss_cost_gate, cost_gate_mean = None, None
    if options['model_name'].endswith('gate'):

        gate_emb = tparams['gate_emb'][ht.flatten()]
        gate_emb = gate_emb.reshape([n_samples, 4, options['edim']])
        sig_gate = tensor.nnet.sigmoid(gate_emb)
        gated_state = sig_gate * ent_emb + (1 - sig_gate) * (
            end_step.reshape((4, n_samples, options['edim']))).dimshuffle([1, 0, 2])
        cost_gate = transe(tparams, options, gated_state[:, 0, :],
                           gated_state[:, 1, :], gated_state[:, 2, :],
                           gated_state[:, 3, :],
                           rel_emb)

        f_loss_cost_gate = theano.function([ht, r, text, mask], outputs=cost_gate, updates=None,
                                           on_unused_input='ignore')
        cost_gate_mean = cost_gate.mean()

    elif options['model_name'].endswith('gate_pro'):
        # n_samples, 4, edim
        txt = (end_step.reshape((4, n_samples, options['edim']))).dimshuffle([1, 0, 2])
        alpha = tensor.nnet.sigmoid(tensor.dot(txt, tparams['gate_W']) +
                                    tensor.dot(ent_emb, tparams['gate_U']) + tparams['gate_b'])

        gated_state = alpha * ent_emb + (1 - alpha) * txt
        cost_gate = transe(tparams, options, gated_state[:, 0, :],
                           gated_state[:, 1, :], gated_state[:, 2, :],
                           gated_state[:, 3, :],
                           rel_emb)

        f_loss_cost_gate = theano.function([ht, r, text, mask], outputs=cost_gate, updates=None,
                                           on_unused_input='ignore')
        cost_gate_mean = cost_gate.mean()
    elif options['model_name'].endswith('gate_s'):
        ###############TODO
        gate_emb = tparams['gate_s'][ht.flatten()]
        gate_emb = gate_emb.reshape([n_samples, 4, 1])
        sig_gate = tensor.nnet.sigmoid(gate_emb)
        # sig_gate = sig_gate[:, :, None]
        gated_state = sig_gate * ent_emb + (1 - sig_gate) * (
            end_step.reshape((4, n_samples, options['edim']))).dimshuffle([1, 0, 2])
        cost_gate = transe(tparams, options, gated_state[:, 0, :],
                           gated_state[:, 1, :], gated_state[:, 2, :],
                           gated_state[:, 3, :],
                           rel_emb)

        f_loss_cost_gate = theano.function([ht, r, text, mask], outputs=cost_gate, updates=None,
                                           on_unused_input='ignore')
        cost_gate_mean = cost_gate.mean()

    # h_pos, t_pos, h_neg, t_neg, r_emb

    # h + r - t
    cost_ori = transe(tparams, options, ent_emb[:, 0, :], ent_emb[:, 1, :], ent_emb[:, 2, :], ent_emb[:, 3, :], rel_emb)

    # h_rnn +r -t
    cost_h_text = transe(tparams, options, h_pos_text, ent_emb[:, 1, :], h_neg_text, ent_emb[:, 3, :], rel_emb)

    # h+r-rnn_t

    cost_t_text = transe(tparams, options, ent_emb[:, 0, :], t_pos_text, ent_emb[:, 2, :], t_neg_text, rel_emb)

    # h_rnn + r - t_rnn
    cost_mul_text = transe(tparams, options, h_pos_text, t_pos_text, h_neg_text, t_neg_text, rel_emb)

    f_loss_cost_ori = theano.function([ht, r, text, mask], outputs=cost_ori, updates=None, on_unused_input='ignore')
    cost_ori_mean = cost_ori.mean()

    f_loss_cost_h_text = theano.function([ht, r, text, mask], outputs=cost_h_text, updates=None,
                                         on_unused_input='ignore')
    cost_h_text_mean = cost_h_text.mean()

    f_loss_cost_t_text = theano.function([ht, r, text, mask], outputs=cost_t_text, updates=None,
                                         on_unused_input='ignore')
    cost_t_text_mean = cost_t_text.mean()

    f_loss_cost_mul_text = theano.function([ht, r, text, mask], outputs=cost_mul_text, updates=None,
                                           on_unused_input='ignore')
    cost_mul_text_mean = cost_mul_text.mean()

    return use_noise, ht, r, text, mask, \
           f_loss_cost_ori, cost_ori_mean, \
           f_loss_cost_h_text, cost_h_text_mean, \
           f_loss_cost_t_text, cost_t_text_mean, \
           f_loss_cost_mul_text, cost_mul_text_mean, \
           f_loss_cost_gate, cost_gate_mean


def build_test(tparams, options, ttparams=None, relation_vec=True):
    if ttparams == None:
        ttparams = tparams
    print 'Test Model: cbow'

    text = tensor.matrix('text', dtype='int64')
    mask = tensor.matrix('text_mask')
    if relation_vec:
        relation = tensor.vector('relation')
    else:
        relation = tensor.matrix('relation')
    #
    # text input shape : seqth_len, batch_size
    #
    # assert text.shape[1] == r.shape[0] * 4
    text_emb = tparams['word_emb'][text.flatten()]
    text_emb = text_emb.reshape([text.shape[0], text.shape[1], options['wdim']])

    if options['wdim'] != options['edim']:
        rt_text = theano.tensor.dot(text_emb, tparams['cbow_W'])
    else:
        rt_text = text_emb
    # rt_text = lstm(tparams, text_emb, options, mask=mask,in_dim=options['wdim'],out_dim=options['edim'])
    # end_step = rt_text[-1]
    # end_step = cbow(tparams, text_emb, options, prefix='cbow', mask=mask)


    if options['model_name'].find('att') != -1:
        end_step = attention(tparams, rt_text, relation, options, prefix='attention', mask=mask)
    else:
        proj = (rt_text * mask[:, :, None]).sum(axis=0)
        end_step = proj / mask.sum(axis=0)[:, None]

    # gate
    alpha = None
    if options['model_name'].endswith('gate'):

        gate_emb = tparams['gate_emb']
        alpha = tensor.nnet.sigmoid(gate_emb)
        gated_state = alpha * ttparams['ent_emb'] + (1 - alpha) * end_step

        end_step = gated_state

    elif options['model_name'].endswith('gate_pro'):
        # n_samples, 4, edim
        txt = end_step
        alpha = tensor.nnet.sigmoid(tensor.dot(txt, tparams['gate_W']) +
                                    tensor.dot(ttparams['ent_emb'], tparams['gate_U']) + tparams['gate_b'])
        gated_state = alpha * ttparams['ent_emb'] + (1 - alpha) * txt

        end_step = gated_state
    elif options['model_name'].endswith('gate_s'):

        gate_emb = tparams['gate_s']
        sig_gate = tensor.nnet.sigmoid(gate_emb)
        gated_state = sig_gate * ttparams['ent_emb'] + (1 - sig_gate) * end_step

        end_step = gated_state

    return text, mask, relation, end_step, alpha


