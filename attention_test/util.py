__author__ = 'jcxu'

from collections import OrderedDict
import theano
import theano.tensor as tensor
import numpy
import cPickle as pkl
import sys
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import config
import cPickle


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


def my_hash(option, h, l, r):
    return h * (option['ent_num'] ** 2) + l * option['ent_num'] + r


def prepare_test_data(new_seq, maxlen=None):
    lengths = [len(seq) for seq in new_seq]

    n_samples = len(new_seq)
    if maxlen is None:
        maxlen = numpy.max(lengths)

    for i, seq in enumerate(new_seq):
        lengths[i] = numpy.minimum(maxlen, lengths[i])
    x = numpy.zeros((maxlen, n_samples), dtype='int64')
    x_mask = numpy.zeros((maxlen, n_samples), dtype='float32')
    # x_mask = numpy.zeros((maxlen, n_samples),dtype='int64')
    for idx, s in enumerate(new_seq):
        x[:lengths[idx], idx] = s[:lengths[idx]]
        x_mask[:lengths[idx], idx] = 1.
    x_mask[0] = 1.
    return x, x_mask


def prepare_data(seqs, maxlen=None):
    """Create the matrices from the datasets.
    @param seqs: a list. shape = (batch_size, 4, seq_len)

    This pad each sequence to the same length: the lenght of the
    longest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    new_seq = []
    for i in range(4):
        for seq in xrange(len(seqs)):
            new_seq.append(seqs[seq][i])
    # new_seq: shape = (seq_len , batch_size * 4)]
    lengths = [len(seq) for seq in new_seq]

    n_samples = len(new_seq)
    if maxlen is None:
        maxlen = numpy.max(lengths)

    for i, seq in enumerate(new_seq):
        lengths[i] = numpy.minimum(maxlen, lengths[i])
    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    # x_mask = numpy.zeros((maxlen, n_samples),dtype='int64')
    for idx, s in enumerate(new_seq):
        x[:lengths[idx], idx] = s[:lengths[idx]]
        x_mask[:lengths[idx], idx] = 1.
    x_mask[0] = 1.
    return x, x_mask


def neg_sample(options, size, samples, train_or_valid='valid'):
    """
    @param size: samples needed
    @param samples: which sample train/valid/test
    #param train_or_valid: a string. 'train' or 'valid'
    @return: [h, l, h', l', r, r'] means: [pos_sample, neg_sample]
    """
    result = []
    # for s in xrange(size / 2):
    for s in xrange(size):
        i = numpy.random.randint(0, len(samples))
        j = numpy.random.randint(0, options['ent_num'])
        k = numpy.random.randint(0, options['rel_num'])
        h, l, r = samples[i][:]
        pr = 1000 * options[train_or_valid + '_right_num'][r] / (
            options[train_or_valid + '_right_num'][r] + options[train_or_valid + '_left_num'][r])
        if options['method'] == 0:
            pr = 500
        if numpy.random.randint(0, 1000) < pr:
            # while triple_count.has_key(h) and triple_count[h].has_key(r) and triple_count[h][r].has_key(j):
            while options[train_or_valid + '_triple_exist'].has_key(my_hash(options, h, j, r)):
                j = numpy.random.randint(0, options['ent_num'])
            result.append([h, l, h, j, r, r])
        else:
            # while triple_count.has_key(j) and triple_count[h].has_key(r) and triple_count[j][r].has_key(l):
            while options[train_or_valid + '_triple_exist'].has_key(my_hash(options, j, l, r)):
                j = numpy.random.randint(0, options['ent_num'])
            result.append([h, l, j, l, r, r])

            # while options[train_or_valid + '_triple_exist'].has_key(my_hash(options, h, l, k)):
            #     k = numpy.random.randint(0, options['rel_num'])
            # result.append([h, l, h, l, r, k])

    x, x_mask = prepare_data([[options['texts'][result[s][i]] for i in xrange(4)] for s in xrange(size)],
                             maxlen=options['max_len'])
    return numpy.array(result), x, x_mask


def generate_test_text(tparams, options, f_test, epo_num, ttparams=None, build_test=None):
    """
    With ATT: options['build_test'] = build_test

    build_test util:
    build_test(tparams, options, ttparams=None, relation_vec=True)
    relation_vec means that relation is a vector, otherwise a matrix.

    Without Att, with gate:  options['test_text_embedding'] = test_text_embedding   options['alpha'] = alpha

    Without Att and gate: options['test_text_embedding'] = test_text_embedding

    :param tparams:
    :param options:
    :param f_test:
    :param epo_num:
    :param ttparams:
    :param build_test: Optional
    :return:
    """
    if ttparams == None:
        ttparams = tparams

    model_name, max_len = options['model_name'], options['max_len']

    # if model_name == 'lstm' or model_name == 'blstm' or model_name == 'cbow'\
    #         or model_name=='lstm_gate' or model_name == 'lstm_gate_pro'\
    #         or model_name=='cbow_gate' or model_name=='cbow_gate_pro':
    #     t_text, t_mask = prepare_test_data(options['texts'], max_len)
    #     test_text_embedding = f_test(t_text, t_mask)[0]

    # if (options['model_name'].find('att') != -1) and (options['data'].find('fb') != -1):
    #     test_num = len(options['test_samples'])
    #     x_left, x_mask_left = prepare_test_data(
    #             [options['texts'][options['test_samples'][s][0]] for s in xrange(test_num)],
    #             maxlen=options['max_len'])
    #
    #     x_right, x_mask_right = prepare_test_data(
    #             [options['texts'][options['test_samples'][s][1]] for s in xrange(test_num)],
    #             maxlen=options['max_len'])
    #     assert len(numpy.sum(x_mask_left, axis=0).nonzero()[0]) == x_mask_left.shape[1]
    #     print 'Pass assertion test'
    #
    #     relation_emb = ttparams['rel_emb'].get_value()
    #     rel_needed = [relation_emb[options['test_samples'][s][2]] for s in xrange(test_num)]
    #
    #     left_test_text_embedding, __ = f_test(x_left, x_mask_left, rel_needed)
    #     right_test_text_embedding, __ = f_test(x_right, x_mask_right, rel_needed)
    #     options['test_text_embedding'] = [left_test_text_embedding, right_test_text_embedding]
    #
    # elif (options['model_name'].find('att') != -1):
    #     t_text, t_mask = prepare_test_data(options['texts'], max_len)
    #
    #     # check zeros mask
    #     assert len(numpy.sum(t_mask, axis=0).nonzero()[0]) == t_mask.shape[1]
    #     print 'Pass assertion test'
    #     relation_emb = ttparams['rel_emb'].get_value()
    #     test_text_embedding = numpy.zeros((options['rel_num'], options['ent_num'], options['edim']))
    #
    #     for i in xrange(options['rel_num']):
    #         tmp_test_text_embedding, __ = f_test(t_text, t_mask, relation_emb[i, :].reshape((options['edim'],)))
    #         test_text_embedding[i] = tmp_test_text_embedding[0]
    #     options['test_text_embedding'] = test_text_embedding

    t_text, t_mask = prepare_test_data(options['texts'], max_len)

    # TODO Attention
    if options['model_name'].find('att')!=-1:
        # if attention, pass the test function
        assert build_test != None
        options['build_test'] = build_test

    # gate
    elif options['model_name'].find('gate') != -1:
        test_text_embedding, alpha = f_test(t_text, t_mask)
        options['test_text_embedding'] = test_text_embedding
        options['alpha'] = alpha


    else:
        test_text_embedding = f_test(t_text, t_mask)
        options['test_text_embedding'] = test_text_embedding


    if ttparams == None:
        pkl.dump([tparams, options], open('t_o_test_%s_%s' % (options['saveto'], str(epo_num)), 'wb'))
    else:
        pkl.dump([ttparams, tparams, options], open('test_%s_%s' % (options['saveto'], str(epo_num)), 'wb'))


def max_norm(p, options):
    s = numpy.square(p)
    s_ = s.sum(axis=1, keepdims=True)
    norms = numpy.sqrt(s_)
    desired = numpy.clip(norms, 0, options['norm'])

    return p * (desired / (1e-7 + norms))
    # norms = K.sqrt(K.sum(K.square(p), axis=0))
    # desired = K.clip(norms, 0, self.m)
    # p = p * (desired / (1e-7 + norms))

def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore')

    return f_grad_shared, f_update



def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore')

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, inputs, cost, lr_option=0.005):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function(inputs, cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    ######
    # idx_ent = tparams.keys().index('ent_emb')
    # idx_rel = tparams.keys().index('rel_emb')
    #
    # tuple_ent = (tparams.get('ent_emb'), tparams.get('ent_emb') - lr_option * lr * gshared[idx_ent])
    # tuple_rel = (tparams.get('rel_emb'), tparams.get('rel_emb') - lr_option * lr * gshared[idx_rel])
    #
    # pup[idx_ent] = tuple_ent
    # pup[idx_rel] = tuple_rel
    list = ['attention_W','attention_U','attention_b','attention_V']
    for i in list:
        idx = tparams.keys().index(i)
        tuple = (tparams.get(i), tparams.get(i) - (1./lr_option) * lr * gshared[idx])
        pup[idx] = tuple
    ######

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adagrad(lr, tparams, grads, inputs, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]

    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]

    rg2up = [(rg2, rg2 + g ** 2) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inputs, cost, updates=zgup + rg2up,
                                    name='adagrad_f_grad_shared')

    updir = [-zg / tensor.sqrt(rg2 + 1e-6)
             for zg, rg2 in zip(zipped_grads,
                                running_grads2)]

    param_up = [(p, p + lr * ud) for p, ud in zip(tparams.values(), updir)]

    # idx = tparams.keys().index('Wemb')
    # param_up.pop(idx)  # Remove the Wemb
    # new_tuple = (tparams.get('Wemb'), tparams.get('Wemb') + 0.1 * updir[idx])
    # param_up.append(new_tuple)

    f_update = theano.function([lr], [], updates=param_up,
                               name='adagrad_f_update')

    return f_grad_shared, f_update


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def ortho_weight(indim, outdim=None):
    if outdim is None:
        W = numpy.random.uniform(low=-.05, high=.05, size=(indim, indim)).astype(config.floatX)
        # W = numpy.random.uniform(
        #     low=-m * numpy.sqrt(6. / (indim + indim)),
        #     high=m * numpy.sqrt(6. / (indim + indim)),
        #     size=(indim, indim)).astype(config.floatX)
    else:
        W = numpy.random.uniform(low=-.05, high=.05, size=(indim, outdim)).astype(config.floatX)
    return W


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


def load_params(path, params):
    pp = numpy.load(path)

    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def new_load_params(path, old_params):
    file = open(path, 'r')
    x = cPickle.load(file)
    if len(x) == 3:
        pp1, pp2, opt = x
        file.close()
        param = old_params

        if type(pp1) == type([]):
            for i in xrange(len(pp1)):
                kk = pp1[i].name
                if param.has_key(kk) == False:
                    print('%s not in model, pass.' % (kk))
                else:
                    print('Reload %s' % (kk))
                    param[kk] = pp1[i].get_value()
        else:
            for kk in pp1:
                if param.has_key(kk) == False:
                    print('%s not in model, pass.' % (kk))
                else:
                    print('Reload %s' % (kk))
                    param[kk] = pp1[kk]

        if type(pp2) == type([]):
            for i in xrange(len(pp2)):
                kk = pp2[i].name
                if param.has_key(kk) == False:
                    print('%s not in model, pass.' % (kk))
                else:
                    print('Reload %s' % (kk))
                    param[kk] = pp2[i].get_value()
        else:
            for kk in pp2:
                if param.has_key(kk) == False:
                    print('%s not in model, pass.' % (kk))
                else:
                    print('Reload %s' % (kk))
                    param[kk] = pp2[kk]

        for i in param:
            print i, type(param[i])
        return param
    elif len(x) == 2:
        par, option = x
        param = old_params
        for kk in par:
            if param.has_key(kk):
                print('%s not in model, pass.' % (kk))
            else:
                print('Reload %s' % (kk))
                param[kk] = par[kk]
        for i in param:
            print i, type(param[i])
        return param


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        try:
            tparams[kk] = theano.shared(params[kk], name=kk)
        except:
            tparams[kk] = params[kk]
        finally:
            print('Load %s in ttp shape:%s\tmean:%s' % (
                kk, tparams[kk].get_value().shape, tparams[kk].get_value().mean()))
    return tparams


def init_tparams_fix(params):
    tparams = OrderedDict()
    ttparams = OrderedDict()
    for kk, pp in params.iteritems():
        if kk == 'ent_emb' or kk == 'rel_emb':
            try:
                ttparams[kk] = theano.shared(params[kk], name=kk)
            except:
                ttparams[kk] = params[kk]
            finally:
                print('Load %s in ttp shape:%s\tmean:%s' % (
                    kk, ttparams[kk].get_value().shape, ttparams[kk].get_value().mean()))
        else:
            try:
                tparams[kk] = theano.shared(params[kk], name=kk)
            except:
                tparams[kk] = params[kk]
            finally:
                print('Load %s in tparam shape:%s\tmean:%s' % (
                    kk, tparams[kk].get_value().shape, tparams[kk].get_value().mean()))
    return tparams, ttparams


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def get_minibatches_idx(n, minibatch_size, shuffle=True):
    """
    Used to shuffle the dataset at each iteration.
    Args:
        n: total length
        minibatch_size: batch size
        shuffle: shuffle data

    Returns:zip(range(len(minibatches)), minibatches)

    """

    idx_list = numpy.arange(n, dtype="int64")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def dropout_layer(state_before, use_noise, trng, noise=0.5):
    proj = tensor.switch(use_noise,
                         state_before * trng.binomial(state_before.shape, p=(1 - noise), n=1, dtype=state_before.dtype),
                         state_before * (1 - noise))
    return proj


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err
