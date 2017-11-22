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

    params = param_init_blstm(options, params, prefix='blstm',in_dim=options['wdim'],out_dim=options['edim'])
    return params


def build_model(tparams, options):
    """

    Args:
        tparams:
        options:

    Returns:

    """
    print 'Model: blstm'
    trng = RandomStreams(817)
    # use_noise = theano.shared(numpy_floatX(0.))
    use_noise = theano.shared(numpy_floatX(0.))
    # head and tail load
    ht = tensor.matrix('ht_triplet', dtype='int64')
    r = tensor.matrix('r_triplet', dtype='int64')
    # assert ht.shape[0] == r.shape[0]

    n_samples = ht.shape[0]
    ent_emb = tparams['ent_emb'][ht.flatten()]
    ent_emb = ent_emb.reshape([n_samples, 4, options['edim']])

    # relation load
    # Naive approach
    rel_emb = tparams['rel_emb'][r.flatten()]
    rel_emb = rel_emb.reshape([n_samples, 2, options['edim']])

    text = tensor.matrix('text', dtype='int64')
    mask = tensor.matrix('text_mask')
    #
    # text input shape : seqth lenth, batch_size*4
    #
    # assert text.shape[1] == r.shape[0] * 4
    text_emb = tparams['word_emb'][text.flatten()]
    text_emb = text_emb.reshape([text.shape[0], text.shape[1], options['wdim']])

    rt_text = blstm(tparams, text_emb, options,prefix='blstm', mask=mask,in_dim=options['wdim'],out_dim=options['edim'])
    end_step = rt_text[-1]

    # assert end_step.shape[0] % 4 == 0
    h_pos_text, t_pos_text, h_neg_text, t_neg_text = end_step[0:r.shape[0]], \
                                                     end_step[r.shape[0]: r.shape[0] * 2], \
                                                     end_step[r.shape[0] * 2:r.shape[0] * 3], \
                                                     end_step[r.shape[0] * 3:r.shape[0] * 4],
    # h_pos, t_pos, h_neg, t_neg, r_emb

    # h + r - t
    cost_ori = transe(tparams, options, ent_emb[:, 0, :], ent_emb[:, 1, :], ent_emb[:, 2, :], ent_emb[:, 3, :], rel_emb)

    # h_rnn +r -t
    cost_h_text = transe(tparams, options, h_pos_text, ent_emb[:, 1, :], h_neg_text, ent_emb[:, 3, :], rel_emb)

    # h+r-rnn_t

    cost_t_text = transe(tparams, options, ent_emb[:, 0, :], t_pos_text, ent_emb[:, 2, :], t_neg_text, rel_emb)

    # h_rnn + r - t_rnn
    cost_mul_text = transe(tparams, options, h_pos_text, t_pos_text, h_neg_text, t_neg_text, rel_emb)

    # cost = transe(tparams, options, ent_emb[:, 0, :], ent_emb[:, 1, :], ent_emb[:, 2, :], ent_emb[:, 3, :], rel_emb) \
    #        + \
    #        transe(tparams, options, h_pos_text, ent_emb[:, 1, :], h_neg_text, ent_emb[:, 3, :], rel_emb) \
    #        + \
    #        transe(tparams, options, ent_emb[:, 0, :], t_pos_text, ent_emb[:, 2, :], t_neg_text, rel_emb) \
    #        + \
    #        transe(tparams, options, h_pos_text, t_pos_text, h_neg_text, t_neg_text, rel_emb)

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
           f_loss_cost_mul_text, cost_mul_text_mean
    #
    # # model.add_input(name='text', input_shape=(None,), dtype='int')    # Assume 4,seq_len
    # model.add_input(name='pos_h_text', input_shape=(None,), dtype='int')
    # model.add_input(name='pos_t_text', input_shape=(None,), dtype='int')
    # model.add_input(name='neg_h_text', input_shape=(None,), dtype='int')
    # model.add_input(name='neg_t_text', input_shape=(None,), dtype='int')
    #
    # model.add_shared_node(Embedding(input_dim=params['vocab_size'],
    #                                 output_dim=params['dim_proj']),
    #                       name='text_embedding',
    #                       inputs=['pos_h_text', 'pos_t_text', 'neg_h_text', 'neg_t_text'],
    #                       outputs=['pos_h_text_emb', 'pos_t_text_emb',
    #                                'neg_h_text_emb', 'neg_t_text_emb']
    #                       )
    #
    # if params['text_encoder'] == 'RNN':
    #     text_encoder = SimpleRNN
    # elif params['text_encoder'] == 'GRU':
    #     text_encoder = GRU
    # else:
    #     text_encoder = LSTM
    #
    # model.add_shared_node(text_encoder(output_dim=params['dim_proj']),
    #                       name='rnn_text_encoder',
    #                       inputs=['pos_h_text_emb', 'pos_t_text_emb', 'neg_h_text_emb', 'neg_t_text_emb'],
    #                       outputs=['pos_h_rep', 'pos_t_rep', 'neg_h_rep', 'neg_t_rep'])

def build_test(tparams, options):
    print 'Test Model: blstm'

    text = tensor.matrix('text', dtype='int64')
    mask = tensor.matrix('text_mask')
    #
    # text input shape : seqth_len, batch_size
    #
    # assert text.shape[1] == r.shape[0] * 4
    text_emb = tparams['word_emb'][text.flatten()]
    text_emb = text_emb.reshape([text.shape[0], text.shape[1], options['wdim']])

    rt_text = blstm(tparams, text_emb, options,prefix='blstm', mask=mask,in_dim=options['wdim'],out_dim=options['edim'])
    end_step = rt_text[-1]
    return text, mask, end_step
