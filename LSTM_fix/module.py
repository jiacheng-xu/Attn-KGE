import numpy, theano
from collections import OrderedDict
import theano
import theano.tensor as tensor
import numpy
import cPickle as pkl
import sys
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import config


# from Util import *

def ortho_weight(indim, outdim=None):
    if outdim is None:
        W = numpy.random.uniform(low=-.2, high=.2, size=(indim, indim)).astype(config.floatX)
    else:
        W = numpy.random.uniform(low=-.2, high=.2, size=(indim, outdim)).astype(config.floatX)
    return W


def _p(pp, name):
    """
    make prefix-appended name
    :param pp: prefix
    :param name: name
    :return: pp_name
    """
    return '%s_%s' % (pp, name)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def param_init_blstm(options, params, prefix='blstm', in_dim=None, out_dim=None):
    """
    Use weights between forward and backward.
    """
    if in_dim is None:
        in_dim = options['wdim']
    if out_dim is None:
        out_dim = options['wdim']

    Wf = numpy.concatenate([ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim)], axis=1)
    params[_p(prefix, 'Wf')] = Wf
    Uf = numpy.concatenate([ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Uf')] = Uf
    bf = numpy.zeros((4 * out_dim,))
    params[_p(prefix, 'bf')] = bf.astype(config.floatX)

    Wb = numpy.concatenate([ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim)], axis=1)
    params[_p(prefix, 'Wb')] = Wb
    Ub = numpy.concatenate([ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Ub')] = Ub
    bb = numpy.zeros((4 * out_dim,))
    params[_p(prefix, 'bb')] = bb.astype(config.floatX)

    Vf = numpy.concatenate([ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Vf')] = Vf
    Vb = numpy.concatenate([ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Vb')] = Vb
    bo = numpy.zeros((out_dim,)).astype(config.floatX)
    params[_p(prefix, 'bo')] = bo
    return params


def blstm(tparams, state_below, options, prefix='blstm', mask=None, in_dim=None, out_dim=None):
    """
    Bidirectional lstm, get the whole h layer
    :param tparams:
    :param state_below: x
    :param options:
    :param prefix:
    :param mask:
    :return: array nsamples, batch_szie, ndim*2
    """
    if out_dim is None:
        out_dim = options['dim_proj']

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'Wf')]) + tparams[_p(prefix, 'bf')]

    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wb')]) + tparams[_p(prefix, 'bb')]

    def _step(m_, x_, h_, c_, U):
        preact = tensor.dot(h_, U)
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, out_dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, out_dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, out_dim))
        c = tensor.tanh(_slice(preact, 3, out_dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h, c

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below_],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim)],
                                non_sequences=[tparams[_p(prefix, 'Uf')]],
                                name=_p(prefix, '_flayers'),
                                n_steps=nsteps)

    bval, updates = theano.scan(_step,
                                sequences=[mask, state_belowx],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim)],
                                non_sequences=[tparams[_p(prefix, 'Ub')]],
                                name=_p(prefix, '_blayers'),
                                go_backwards=True,
                                n_steps=nsteps)
    #####
    #
    rt_fwd = rval[0]  # h of forward step
    rt_bwd = bval[0][::-1, :, :]  # h of backward step, and reverse in the axis=0

    ret_h = tensor.dot(rt_fwd, tparams[_p(prefix, 'Vf')]) + tensor.dot(rt_bwd, tparams[_p(prefix, 'Vb')]) + tparams[
        _p(prefix, 'bo')]

    # Like 0,1,2    3,4,5
    #      3,4,5    0,1,2
    #
    ####
    # 97,16,128  *  2   ==> 97,16,256
    # sum
    # rt = concatenate([rval[0], bval[0]], axis=rval[0].ndim - 1)  # 97,16, 256
    # end
    # rt = concatenate([rval[0][-1], bval[0][-1]], axis=rval[0][-1].ndim - 1)  # 16, 256

    # rt = [rval[0], bval[0]]
    return ret_h


def param_init_lstm(options, params, prefix='lstm', in_dim=None, out_dim=None):
    if in_dim is None:
        in_dim = options['wdim']
    if out_dim is None:
        out_dim = options['wdim']

    W = numpy.concatenate([ortho_weight(in_dim, out_dim),
                           ortho_weight(in_dim, out_dim),
                           ortho_weight(in_dim, out_dim),
                           ortho_weight(in_dim, out_dim)], axis=1)
    params[_p(prefix, 'W')] = W.astype(config.floatX)
    U = numpy.concatenate([ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'U')] = U.astype(config.floatX)
    b = numpy.zeros((4 * out_dim,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)
    return params


def lstm(tparams, state_below, options, prefix='lstm', mask=None, in_dim=None, out_dim=None):
    if out_dim is None:
        out_dim = options['wdim']

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, out_dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, out_dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, out_dim))
        c = tensor.tanh(_slice(preact, 3, out_dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = out_dim
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    # TODO
    return rval[0]


def param_init_attention(options, params, prefix='attention', dim=None):
    if dim is None:
        dim = options['edim']

    W = ortho_weight(dim)
    params[_p(prefix, 'W')] = W

    U = ortho_weight(dim)
    params[_p(prefix, 'U')] = U

    b = numpy.zeros((dim,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    V = numpy.random.uniform(low=-0.2, high=0.2, size=(dim,)).astype(config.floatX)
    params[_p(prefix, 'V')] = V

    # bb = numpy.zeros((1,))
    # params[_p(prefix, 'bb')] = bb.astype(config.floatX)


    return params


def attention(tparams, state_below, rel, options, prefix='attention', mask=None):


    # fix nan

    # pctx = 97,16,256  256*256  + 256, = 97,16,256

    pctx = tensor.dot(state_below, tparams[_p(prefix, 'W')]) \
           + tensor.dot(rel, tparams[_p(prefix, 'U')])\
           + tparams[_p(prefix, 'b')]
    # pctx = state_below



    pctx_ = tensor.tanh(pctx)
    # alpha = theano.tensor.sum(pctx_,axis=2)

    # x, = input_storage
    # e_x = numpy.exp(x - x.max(axis=1)[:, None])
    #  sm = e_x / e_x.sum(axis=1)[:, None]
    #  output_storage[0][0] = sm


    # alpha = 97,16,256 * 256,  = 97,16
    alpha = tensor.dot(pctx_, tparams[_p(prefix, 'V')])
    rt = alpha


    e_x = theano.tensor.exp(alpha - alpha.max(axis=0)[None,:])

    e_x_mask = e_x * mask
    alpha = e_x_mask / theano.tensor.sum(e_x_mask, axis=0, keepdims=True)

    f_att = theano.function(inputs=[state_below, rel, mask], outputs=[alpha],name='att_monitor')
    options['f_att'] = f_att
    # alpha = tensor.nnet.sigmoid(alpha)
    # alpha = alpha * mask
    # alpha = alpha / theano.tensor.sum(alpha, axis=0, keepdims=True)
    # alpha.sum(axis=0)
    # h = emb * alpha[:, :, None]
    # h = tensor.dot(state_below,alpha)
    # h = state_below * alpha[:, :, None]
    # alpha


    # 97,16,1 * 97,16,256
    proj = state_below
    # proj = alpha[:, :, None] * state_below



    # proj = (h * mask[:, :, None]).sum(axis=0)
    # proj = proj / mask.sum(axis=0)[:, None]
    # proj = tensor.tanh(tensor.dot(proj, tparams[_p(prefix, 'O')]))

    # h is 97,16,128

    # def _step(m_, x_, h_):
    #     h = m_[:, None] * x_ + (1. - m_)[:, None] * h_
    #     return h
    #
    # rval, updates = theano.scan(_step,
    #                             sequences=[mask, state],
    #                             outputs_info=[tensor.alloc(numpy_floatX(0.),
    #                                                        n_samples,
    #                                                        options['edim'])],
    #                             name='attention_mask')
    # return rval[-1]

    # proj = state[-1]

    # proj = (proj * mask[:, :, None]).sum(axis=0)
    # proj = proj / mask.sum(axis=0)[:, None]


    return proj,rt




def param_init_cbow(options, params, prefix='cbow', in_dim=None, out_dim=None):
    # W = ortho_weight(indim=in_dim, outdim=out_dim)
    # params[_p(prefix, 'W')] = W
    return params


# def cbow(tparams, state_below, options, prefix='cbow', mask=None, in_dim=None, out_dim=None):
#     assert mask is not None
#     mask_sum = theano.tensor.sum(mask, axis=0)
#     x = state_below * mask[:, :, None]
#     c = theano.tensor.sum(x, axis=0)
#     d = mask_sum.reshape((mask_sum.shape[0], 1))
#     ent = c / d
#     rt = theano.tensor.dot(ent, tparams[_p(prefix, 'W')])
    # return ent

def cbow(tparams, state_below, options, prefix='cbow', mask=None, in_dim=None, out_dim=None):
    assert mask is not None
    mask_sum = theano.tensor.sum(mask, axis=0)
    new_state = state_below * mask[:, :, None]
    c = theano.tensor.sum(new_state, axis=0)
    d = mask_sum.reshape((mask_sum.shape[0], 1))
    ent = c / d
    ent_sigmoid = tensor.nnet.sigmoid(ent)
    return ent_sigmoid