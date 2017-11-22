import os, numpy, theano, time
import cPickle as pkl
from util import *
import getopt
import sys
# from lstm_launch import *
# from lstm_att_launch import *

from theano import config


def handle_argv(argv):
    opts, args = getopt.getopt(argv[1:], 'm:d:l:n:p:s:',
                               ['marg=', 'dist=', 'len=', 'name=', 'load=', 'batch=', 'loss=', 'edim=', 'wdim=', 'lr=',
                                'lrtxt='])
    print opts
    margin = 2.0
    distance = 'l1'
    max_len = 20
    model = 'lstm'
    batch_size = 32
    load_param = '0111_fb_100_50_cbow_l1_2.0_20_1.5_sgd.pkl'
    loss_code = '01110'
    edim = 100
    wdim = 50
    lr = 0.05
    lrtxt = 'lrate'
    for o, a in opts:
        if o == '--marg':
            margin = float(a)
        elif o == '--dist':
            distance = a
        elif o == '--len':
            max_len = int(a)
        elif o == '--name':
            model = a
        elif o == '--load':
            load_param = a
        elif o == '--batch':
            batch_size = int(a)
        elif o == '--loss':
            loss_code = a
        elif o == '--edim':
            edim = a
        elif o == '--wdim':
            wdim = a
        elif o == '--lr':
            lr = float(a)
        elif o == '--lrtxt':
            lrtxt = a

    train(margin=margin, distance=distance, max_len=max_len,
          model_name=model, load_param=load_param, batch_size=batch_size,
          loss_code=loss_code, edim=edim, wdim=wdim,
          lrate=lr, lrtxt=lrtxt)


def train(
        data='fb_100_50', model_name='lstm', load_param=None,
        loss_code='0111',
        distance='l1',
        norm=1,
        batch_size=64,
        max_len=None,
        margin=2.0,
        edim=20,
        wdim=50,
        lrtxt='lrate.txt',
        decay_c=1e-9,
        optimizer=sgd,
        valid_batch_size=128,
        disp_freq=2000,
        valid_freq=50000,
        save_freq=20000,
        lrate=0.07,
        max_epochs=3000, patience=1000,
        use_noise=0., noise=0.5,
        encoder='lstm',
        method=1
):
    options = locals().copy()
    print(options)

    def process_loss_code(loss_code, digit, options):
        if loss_code[digit] == '1':
            options['loss%s' % (digit)] = True
        else:
            options['loss%s' % (digit)] = False
        return options

    options = process_loss_code(loss_code, 0, options)
    options = process_loss_code(loss_code, 1, options)
    options = process_loss_code(loss_code, 2, options)
    options = process_loss_code(loss_code, 3, options)
    options = process_loss_code(loss_code, 4, options)

    ##################
    print 'Loading data'

    saveto = ('TC_%s%s%s%s%s_%s_%s_%s_%s_%s_%s_%s.pkl') % (
        int(options['loss0']), int(options['loss1']), int(options['loss2']),
        int(options['loss3']),int(options['loss4']), data, model_name, distance, margin,
        max_len, lrate, optimizer.__name__)

    options['saveto'] = saveto
    print '-------------SAVE TO PATH: %s--------------' % (saveto)

    if model_name == 'lstm_att':
        from lstm_att_launch import transe, build_model, init_params, build_test
    if model_name == 'lstm':
        from lstm_launch import  build_model, init_params, build_test
    elif model_name == 'cbow':
        from cbow_launch import transe, build_model, init_params, build_test
    else:
        raise NotImplementedError

    path = os.path.join('..', 'data', data + '.pkl')
    data = pkl.load(open(path, 'rb'))

    # TODO
    options['ent_num'] = data['entity_num']
    options['rel_num'] = data['relation_num']
    options['all_triple_exist'] = data['all_triple_exist']
    options['test_samples'] = data['test_samples']
    options['texts'] = data['texts']
    options['train_left_num'] = data['train_left_num']
    options['train_right_num'] = data['train_right_num']
    options['train_samples'] = data['train_samples']
    options['train_triple_exist'] = data['train_triple_exist']
    options['valid_left_num'], options['valid_right_num'], options['valid_samples'], \
    options['valid_triple_exist'] = data['valid_left_num'], data['valid_right_num'], data['valid_samples'], data[
        'valid_triple_exist']

    print 'Building model'
    params = init_params_tc(options)

    word_emb = data['word_emb']
    params['word_emb'] = word_emb.astype(config.floatX)
    # params['word_emb'] = numpy.random.uniform(-.1, .1, size=(word_emb.shape[0], options['edim'])).astype(config.floatX)

    params['ent_emb'] = data['entity_emb'].astype(config.floatX)
    params['rel_emb'] = data['relation_emb'].astype(config.floatX)
    assert params['ent_emb'].shape[0] == options['ent_num']

    if load_param is not None:
        try:
            params = new_load_params(load_param, params)
            print 'load file succ: ' + load_param
        except ImportError:
            print('cuda not found')
        except Exception, e:
            import traceback
            traceback.print_exc()
            print 'load file fail:' + load_param

    tparams, ttparams = init_tparams_fix(params)

    # params['ent_emb'] = numpy.random.uniform(-.4, .4, size=(options['ent_num'], 20)).astype(config.floatX)
    # params['rel_emb'] = numpy.random.uniform(-.4, .4, size=(options['rel_num'], 20)).astype(config.floatX)
    # tparams['word_emb'].set_value(max_norm(tparams['word_emb'].get_value(), options))

    #####
    # Prepare test data
    if model_name == 'lstm' or model_name == 'lstm_gate' or model_name == 'cbow' or model_name == 'lstm_gate_pro' or model_name=='cbow_gate' or model_name=='cbow_gate_pro':
        test_text, test_mask, test_end_step = build_test(tparams, options, ttparams)
        f_test = theano.function(inputs=[test_text, test_mask], outputs=[test_end_step])
    elif model_name == 'lstm_att' or model_name == 'lstm_att_gate' or model_name=='lstm_att_gate_pro':
        test_text, test_mask, rel, test_end_step = build_test(tparams, options, ttparams)
        f_test = theano.function(inputs=[test_text, test_mask, rel], outputs=[test_end_step])
    ######

    if model_name == 'transe':
        (ht, r, f_loss_cost_ori, cost_ori) = build_model(tparams, options)
    elif model_name == 'lstm_gate' or model_name == 'lstm_att_gate' or model_name == 'cbow_gate' or model_name == 'lstm_gate_pro' or model_name == 'lstm_att_gate_pro' or model_name == 'cbow_gate_pro':
        (use_noise, ht, r, text, mask,
         f_loss_cost_ori, cost_ori,
         f_loss_cost_h_text, cost_h_text,
         f_loss_cost_t_text, cost_t_text,
         f_loss_cost_mul_text, cost_mul_text,
         f_loss_cost_gate, cost_gate) = build_model(tparams, options, ttparams)
    elif model_name == 'lstm' or model_name == 'lstm_att' or model_name == 'cbow':
        (use_noise, ht, r, text, mask,
         f_loss_cost_ori, cost_ori,
         f_loss_cost_h_text, cost_h_text,
         f_loss_cost_t_text, cost_t_text,
         f_loss_cost_mul_text, cost_mul_text) = build_model(tparams, options, ttparams)
    else:
        raise NotImplementedError

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        # TODO
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c

        if encoder == 'transe':
            cost_ori += weight_decay
        else:
            if options['loss0'] == True:
                cost_ori += weight_decay
            elif options['loss1'] == True:
                cost_h_text += weight_decay
            elif options['loss2'] == True:
                cost_t_text += weight_decay
            elif options['loss3'] == True:
                cost_mul_text += weight_decay
            elif options['loss4'] == True:
                cost_gate += weight_decay




    lr = tensor.scalar(name='lr')
    if options['loss0'] == True:
        f_cost_ori = theano.function([ht, r], cost_ori, name='f_cost_ori')
        grads_ori = tensor.grad(cost_ori, wrt=tparams.values())
        f_grad_ori = theano.function([ht, r], grads_ori, name='f_grad_ori')
        f_grad_shared_ori, f_update_ori = optimizer(lr, tparams, grads_ori,
                                                    [ht, r], cost_ori)
    if options['loss1'] == True:
        f_cost_h_text = theano.function([ht, r, text, mask], cost_h_text, name='f_cost_h_text')
        grads_h_text = tensor.grad(cost_h_text, wrt=tparams.values())
        f_grad_h_text = theano.function([ht, r, text, mask], grads_h_text, name='f_grad_h_text')
        f_grad_shared_h_text, f_update_h_text = optimizer(lr, tparams, grads_h_text,
                                                          [ht, r, text, mask], cost_h_text)
    if options['loss2'] == True:
        f_cost_t_text = theano.function([ht, r, text, mask], cost_t_text, name='f_cost_t_text')
        grads_t_text = tensor.grad(cost_t_text, wrt=tparams.values())
        f_grad_t_text = theano.function([ht, r, text, mask], grads_t_text, name='f_grad_t_text')
        f_grad_shared_t_text, f_update_t_text = optimizer(lr, tparams, grads_t_text,
                                                          [ht, r, text, mask], cost_t_text)
    if options['loss3'] == True:
        f_cost_mul_text = theano.function([ht, r, text, mask], cost_mul_text, name='f_cost_mul_text')
        grads_mul_text = tensor.grad(cost_mul_text, wrt=tparams.values())
        f_grad_mul_text = theano.function([ht, r, text, mask], grads_mul_text, name='f_grad_mul_text')
        f_grad_shared_mul_text, f_update_mul_text = optimizer(lr, tparams, grads_mul_text,
                                                              [ht, r, text, mask], cost_mul_text)
    if options['loss4'] == True:
        f_cost_gate = theano.function([ht, r, text, mask], cost_gate, name='f_cost_gate')
        grads_gate = tensor.grad(cost_gate, wrt=tparams.values())
        f_grad_gate = theano.function([ht, r, text, mask], grads_gate, name='f_grad_gate')
        f_grad_shared_gate, f_update_gate = optimizer(lr, tparams, grads_gate,
                                                      [ht, r, text, mask], cost_gate)

    print 'Optimization'

    print "%d train examples" % (options['train_samples'].shape[0])
    print "%d valid examples" % (options['valid_samples'].shape[0])
    print "%d test examples" % (options['test_samples'].shape[0])

    history_errs = []
    best_p = None
    bad_count = 0

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in xrange(max_epochs):

            n_samples = 0

            batch_num = int(options['train_samples'].shape[0] / batch_size)
            total_loss = [0, 0, 0, 0, 0]
            for i in xrange(batch_num):

                uidx += 1
                triple, text, text_mask = neg_sample(options, batch_size, options['train_samples'], 'train')

                use_noise.set_value(1.)

                n_samples += batch_size

                if encoder == 'transe':
                    cost_ori = f_grad_shared_ori(triple[:, :4], triple[:, 4:6])
                    f_update_ori(lrate)
                elif encoder == 'lstm':
                    if options['loss0'] == True:
                        cost_ori = f_grad_shared_ori(triple[:, :4], triple[:, 4:6])
                        f_update_ori(lrate)
                        total_loss[0] += cost_ori
                    if options['loss1'] == True:
                        cost_h_text = f_grad_shared_h_text(triple[:, :4], triple[:, 4:6], text, text_mask)
                        f_update_h_text(lrate)
                        total_loss[1] += cost_h_text
                    if options['loss2'] == True:
                        cost_t_text = f_grad_shared_t_text(triple[:, :4], triple[:, 4:6], text, text_mask)
                        f_update_t_text(lrate)
                        total_loss[2] += cost_t_text
                    if options['loss3'] == True:
                        cost_mul_text = f_grad_shared_mul_text(triple[:, :4], triple[:, 4:6], text, text_mask)
                        f_update_mul_text(lrate)
                        total_loss[3] += cost_mul_text
                    if options['loss4'] == True:
                        cost_gate = f_grad_shared_gate(triple[:, :4], triple[:, 4:6], text, text_mask)
                        f_update_gate(lrate)
                        total_loss[4] += cost_gate
                # tparams['rel_emb'].set_value(max_norm(tparams['rel_emb'].get_value(), options))
                # tparams['ent_emb'].set_value(max_norm(tparams['ent_emb'].get_value(), options))

                if numpy.mod(uidx, disp_freq) == 0:
                    if options['loss0']:
                        print 'Epoch\t', eidx, 'Update\t', uidx, 'Cost_o\t', cost_ori
                    if options['loss1']:
                        print 'Epoch\t', eidx, 'Update\t', uidx, 'Cost_h\t', cost_h_text
                    if options['loss2']:
                        print 'Epoch\t', eidx, 'Update\t', uidx, 'Cost_t\t', cost_t_text
                    if options['loss3']:
                        print 'Epoch\t', eidx, 'Update\t', uidx, 'Cost_m\t', cost_mul_text
                    if options['loss4']:
                        print 'Epoch\t', eidx, 'Update\t', uidx, 'Cost_g\t', cost_gate, ' alpha mean\t', (
                        tparams['gate_emb'].get_value()).mean()
                if saveto and numpy.mod(uidx, save_freq) == 0:
                    print 'Saving...',
                    pkl.dump([ttparams, tparams, options], open('%s' % saveto, 'wb'), -1)
                    print 'Done'

                if numpy.mod(uidx, 2 * save_freq) == 0:
                    print('--SAVING TEST--')
                    generate_test_text(tparams, options, f_test, eidx, ttparams)

            print 'Seen %d samples' % n_samples
            print 'Total Loss:\t%s Epo:%s' % (total_loss, eidx)

            # Dynamically change learning rate
            try:
                with open('%s.txt' % (lrtxt), 'r') as f:
                    l = float(f.readline())
                    if l < 0.:
                        if total_loss > 200:
                            lrate = 0.1
                        elif total_loss > 100:
                            lrate = 0.05
                        elif total_loss > 50:
                            lrate = 0.008
                        else:
                            lrate = 0.005
                    else:
                        lrate = l
                    print('LrFile %s.txt loaded. Lrate=%s' % (lrtxt, str(lrate)))
            except EOFError:
                print('LrFile %s.txt Not found.' % (lrtxt))
            except:
                print('load error!')

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    generate_test_text(tparams, options, f_test, 9999, ttparams)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))


handle_argv(sys.argv)
