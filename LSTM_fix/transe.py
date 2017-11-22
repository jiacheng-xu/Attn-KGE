from util import *


def cal_distance(tparams, options):
    pass

def init_params(model_options):
    params = OrderedDict()
    return params
def transe(tparams, options, ht_emb, r_emb):
    if options['distance'] == 'l1':
        return options['margin'] + tensor.sum(tensor.abs_(ht_emb[:, 0, :] + r_emb[:, 0, :] - ht_emb[:, 1, :]), axis=1) - \
               tensor.sum(tensor.abs_(ht_emb[:, 2, :] + r_emb[:, 1, :] - ht_emb[:, 3, :]), axis=1)
    elif options['distance'] == 'l2':
        return options['margin'] + tensor.sum(tensor.sqr(ht_emb[:, 0, :] + r_emb[:, 0, :] - ht_emb[:, 1, :]), axis=1) - \
               tensor.sum(tensor.sqr(ht_emb[:, 2, :] + r_emb[:, 1, :] - ht_emb[:, 3, :]), axis=1)
    else:
        raise NotImplementedError('Illegal distance measure.')


def build_model(tparams, options):
    trng = RandomStreams(817)
    # use_noise = theano.shared(numpy_floatX(0.))

    # head and tail load
    ht = tensor.imatrix('ht_triplet')
    r = tensor.imatrix('r_triplet')

    # assert ht.shape[0] == r.shape[0]
    n_samples = ht.shape[0]
    ent_emb = tparams['ent_emb'][ht.flatten()]
    ent_emb = ent_emb.reshape([n_samples, 4, options['dim_proj']])

    # relation load
    # Naive approach
    rel_emb = tparams['rel_emb'][r.flatten()]
    rel_emb = rel_emb.reshape([n_samples, 2, options['dim_proj']])

    # TODO text
    cost = transe(tparams, options, ent_emb, rel_emb)

    f_loss = theano.function([ht, r], outputs=cost, updates=None)
    cost_mean = cost.mean()

    return ht, r, f_loss, cost_mean
