from model.efm import *
from model.lrppm import *
import tensorflow as tf
from data_loader import DataLoader
import numpy as np
import pandas as pd
import pdb
import xktensorflow as xktf
import time

'''
Solve a model: 
1. train the model
2. evaluate the model
'''


class Solver:
    def __init__(self, args):
        print '[INFO] Start Solver'
        self.args = args
        self.data_loader = self.override_returnDataLoader()(args)
        self.data_loader.get_alldata_putin_dic(args)
        start_time = time.time()
        self.override_build_model()
        self.run()
        print '[STATISTIC] time usage : %s ms' % ((time.time() - start_time) * 1000)
        print self.accurate()
        print self.override_predict(0) #the 0 user 's recommendation
        pass

    def accurate(self) :  # normal loss
        # return [ precision , recall ]
        print '[INFO] Start Loss Calculate'
        d_u_stestm = self.args['d_u_stestm']
        prec = 0.0
        prec_tot = 0.0
        recall = 0.0 
        recall_tot = 0.0
        for t_u , t_sm in d_u_stestm.items() : 
            s_predict = set(self.override_predict(t_u))
            assert(isinstance(s_predict , set))
            prec += len (s_predict.intersection(t_sm) )
            prec_tot += len(s_predict)
            recall += len (s_predict.intersection(t_sm) )
            recall_tot +=  len(t_sm)
        print 'DEBUG' , len(d_u_stestm) , prec , prec_tot
        prec /= prec_tot
        recall /= recall_tot
        return [prec , recall]

    def override_returnDataLoader(self) :
        print '[ERROR] please override_returnDataLoader !!'
        exit(-1)
    def override_predict(self , usr) : 
        print '[WARN ] default predict routine'
        exit(-1)
    def override_pre_predict(self) : #calculate the nessessary from tensorflow
        print '[WARN ] default predict routine'
        exit(-1)

    #must add self.train = tf.ops
    def override_build_model(self ) : 
        pass

    #must make sure you put 'solver_train_set' in the args
    def override_pre_run(self) :  
        pass
    def override_post_run(self) : 
        pass
    #call when a epoch has end
    def override_post_epoch(self) : #POST epoch
        pass

    def run(self) : 
        self.tfsession = tf.Session()

        self.override_pre_run()  
        self.override_run()
        self.override_post_run()

    def override_run(self): # normal deep learning style run 
        # call the self.train operation
        # call the self.
        args = self.args
        s = self.tfsession
        if ('solver_train_set' not in self.args ) : 
            print '[ERROR] solver_train_set not found'
            exit (-1)

        train_set = args['solver_train_set']
        s.run(tf.global_variables_initializer())
        batch_size = int(args['batch_size'])
        num_step  = (len(train_set)+batch_size-1) // batch_size 
        alpha = float(args['learn_rate'])
        print '[TRAIN] max_epoch = %s\t\tnum_step = %s' % (args['epoch'] , num_step)
        for epoch in range(int(args['epoch'])) : 
            np.random.shuffle(train_set)
            tot_loss = 0.0
            for step in range(num_step): 
                end = batch_size*(step+1) if step < num_step-1 else len(train_set) + 1
                feed_data = {self.batch_set : train_set[step*batch_size:end] , self.learn_rate : alpha}
                if args.get('debug' , 'false') == 'true' : 
                    import pdb
                    pdb.set_trace()

                #import pdb 
                #pdb.set_trace()
                # add your code here
                s.run(self.train , feed_dict=feed_data)
                # end your code 
                #import pdb 
                #pdb.set_trace()

                if( self.args.get('shut_down_loss' , 'false') == 'true' ) : 
                    cnt_loss = 0
                else : 
                    cnt_loss = s.run(self.loss , feed_dict=feed_data)
                if( not step % 100 and self.args.get('echo_step' , 'false') == 'true') : 
                    print '[TRAIN epoch %s , step %s] loss = %s'  % ( epoch \
                        , step , cnt_loss)
                tot_loss += cnt_loss
            self.override_post_epoch()
            accurate = 'WAIT'
            if not (epoch+1) % int(args['echo_accurate_interval_epoch']) : 
                self.override_pre_predict()
                accurate = self.accurate()
            print '[TRAIN epoch %s] Loss = %s\t\tAccurate = %s'%(epoch , tot_loss \
                , accurate)
            alpha *= 0.9
