# import numpy as np
#
# x = np.arange(60).reshape((3,4,5))
# mask = np.ones((3,4))
# mask[2][3] = 0
# mask[1][3] = 0
#
# W = np.ones((5,5))
# U = np.ones((5,5))
# rel = np.ones((5,))
#
#
#     pctx = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tensor.dot(rel, tparams[_p(prefix, 'U')]) + tparams[
#         _p(prefix, 'b')]
#
#
#     pctx_ = tensor.tanh(pctx)
#
#
#     # seq, batch, dim *
#     # alpha = 97,16,256 * 256,  = 97,16
#     alpha = tensor.dot(pctx_, tparams[_p(prefix, 'V')])
#     alpha = tensor.exp(alpha)
#     alpha = alpha * mask
#     alpha = alpha / theano.tensor.sum(alpha, axis=0, keepdims=True)
#     # alpha.sum(axis=0)
#     # h = emb * alpha[:, :, None]
#     # h = tensor.dot(state_below,alpha)
#     # h = state_below * alpha[:, :, None]
#     # alpha
#     state = alpha[:, :, None] * state_below
#     # proj = (h * mask[:, :, None]).sum(axis=0)
#     # proj = proj / mask.sum(axis=0)[:, None]
#     # proj = tensor.tanh(tensor.dot(proj, tparams[_p(prefix, 'O')]))
#
#     # h is 97,16,128
#
#     # def _step(m_, x_, h_):
#     #     h = m_[:, None] * x_ + (1. - m_)[:, None] * h_
#     #     return h
#     #
#     # rval, updates = theano.scan(_step,
#     #                             sequences=[mask, state],
#     #                             outputs_info=[tensor.alloc(numpy_floatX(0.),
#     #                                                        n_samples,
#     #                                                        options['edim'])],
#     #                             name='attention_mask')
#     # return rval[-1]
#     proj = (state * mask[:, :, None]).sum(axis=0)
#     proj = proj / mask.sum(axis=0)[:, None]
#     return proj
#
