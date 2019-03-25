from data_loader import DataLoader
from data_loader import CacheFile
import tensorflow as tf
from solver import Solver
import numpy as np
import random
class CommonCacheData(CacheFile) : 
    #@Override 
    def override_get_file_list_(self) : 
        return ['a_trainset' , 'a_testset' , 'd_u_stestm' , 'd_u_strainm' , 'd_um_r']
    def override_generate_data(self , dic_output) :
        l_dataset = []
        M = self.data_loader.M
        print '[DATA] file path : ','%s/u.data'%self.args['base_path']
        fp_r = open('%s/u.data'%self.args['base_path'] , 'r')
        for line in fp_r.readlines() : 
            t_u = int(line.strip().split()[0]) - 1
            t_m = int(line.strip().split()[1]) - 1
            t_r = int(line.strip().split()[2])
            assert(t_u >= 0)
            assert(t_m >= 0)
            assert(t_r >= 0)
            l_dataset.append([t_u , t_m , t_r])

        a_dataset = (np.array(l_dataset))
        np.random.shuffle(a_dataset)
        n_dataset = len(a_dataset) 
        n_trainset= n_dataset / M * (M-1)
        n_testset = n_dataset - n_trainset
        
        a_trainset = a_dataset[:n_trainset]
        a_testset = a_dataset[n_trainset:]
        d_u_stestm = dict()
        d_u_strainm = dict()
        d_um_r = dict()
        for t_u , t_m , t_r in a_trainset : 
            if t_u not in d_u_strainm : 
                d_u_strainm[t_u] = set()
            d_u_strainm[t_u].add(t_m)
            d_um_r[(t_u , t_m)] = t_r

        for t_u , t_m , t_r in a_testset : 
            if t_u not in d_u_stestm : 
                d_u_stestm[t_u] = set()
            d_u_stestm[t_u].add(t_m)

        dic_output['a_trainset'] = a_trainset
        dic_output['d_u_stestm'] = d_u_stestm
        dic_output['a_testset'] = a_testset
        dic_output['d_u_strainm'] = d_u_strainm
        dic_output['d_um_r'] = d_um_r
        

class LFM_CacheData_umr(CacheFile):
    #@Override 
    def override_get_file_list_(self) : 
        return ['a_umr']
    def override_generate_data(self , dic_output) :
        d_dataset = self.data_loader.get_list_return_dict(['a_trainset' , 'a_testset'])
        d_u_sm_pos = {}
        d_u_sm_neg = {}
        for tu , tm , tr in d_dataset['a_trainset'] : 
            if tu not in d_u_sm_pos : d_u_sm_pos[tu] = set()
            d_u_sm_pos[tu].add(tm)

        for tu , tsm in d_u_sm_pos.items() : 
            maxx = len(tsm)
            tmpset = set()
            for i in range(3*maxx) : 
                ran = random.randint(0 , self.data_loader.i_mm-1)
                if ran in tsm or ran in tmpset : 
                    continue
                tmpset.add(ran)
                if len(tmpset) > maxx : break
            d_u_sm_neg[tu] = tmpset
                
        l_umr = []
        d_um_r = self.args['d_um_r']
        for tu , tsm in d_u_sm_pos.items() : 
            for tm in tsm : 
                l_umr.append([tu , tm , 1.0])
        for tu , tsm in d_u_sm_neg.items() : 
            for tm in tsm : 
                l_umr.append([tu , tm , 0.0])

        a_umr = np.array(l_umr) 
        dic_output['a_umr'] = a_umr


class LFMdata_loader(DataLoader) : 
    #@Override
    def override_register_all(self) : 
        self.register(CommonCacheData)
        self.register(LFM_CacheData_umr)

    #@Override
    def override_option_args_process(self) : 
        args = self.args
        self.i_mu = int(args['max_user'])
        self.i_mm = int(args['max_item'])
            
        self.M = int(args['m'])

class LFMSolver (Solver): 
    def override_build_model(self) : 
        args = self.args
        with tf.device('/cpu:0')  : 
            initializer = tf.random_uniform_initializer(0 , 1) ;     
            self.p_u = p_u = tf.get_variable('p_u' , shape=[int(args['max_user']) , int(args['lfm_r'])] , initializer=initializer)
            self.p_m = p_m = tf.get_variable('p_m' , shape=[int(args['max_item']) , int(args['lfm_r'])] , initializer=initializer)

            self.batch_set = tf.placeholder('int32' , [None , 3])
            self.learn_rate = tf.placeholder('float32')
            self.vec_usr = vec_usr = tf.nn.embedding_lookup(p_u ,self.batch_set[:,0])
            self.vec_mov = vec_mov = tf.nn.embedding_lookup(p_m ,self.batch_set[:,1])
            self.vec_rat = vec_rat = tf.cast(tf.reshape(self.batch_set[:,2] , [-1,1]) , 'float32')

            tmp_loss = (tf.reduce_sum(tf.multiply(vec_usr , vec_mov) , axis=-1) - vec_rat)**2 
            loss = tf.reduce_sum(tmp_loss) + float(args['lambda'])*(tf.nn.l2_loss(vec_mov) + tf.nn.l2_loss(vec_usr))
            #global_steps = tf.Variable(0 , dtype=tf.int32 , trainable=False)
            #self.learning_rate = tf.train.exponential_decay(float(args['learn_rate']) , global_steps , 2000 , 0.7)
            opt = tf.train.GradientDescentOptimizer(self.learn_rate)
            opt = opt.minimize(loss)
            self.pre_predict = tf.matmul(self.p_u , tf.transpose(self.p_m))
            self.train = opt
            self.loss  = loss
            self.maxx  = tf.reduce_max(tf.concat([self.p_u,self.p_m] , axis=0))


    def override_returnDataLoader(self) : 
        print '[DATA] Get the LFMdata_loader'
        return LFMdata_loader
        
    def override_pre_run(self) :
        self.args['solver_train_set'] = self.args['a_umr']
    def override_post_run(self) : 
        pass
    def override_pre_predict(self) : 
        a_u_m_top_k = self.tfsession.run(self.pre_predict)
        d_u_lrankm = {}
        for t_u , t_lm in enumerate(a_u_m_top_k) : 
            li = []
            for i in enumerate(t_lm) : 
                li.append(i)

            d_u_lrankm [t_u] = sorted(li , key=lambda d:d[1] , reverse=True)
        #print 'list' , d_u_lrankm[0]
        #import pdb 
        #pdb.set_trace()
        self.args['d_u_lrankm'] = d_u_lrankm
        # print '[DEBUG]' , self.tfsession.run(self.maxx)

    def override_predict(self , usr) : 
        l_rank = self.args['d_u_lrankm'][usr]
        s_train = self.args['d_u_strainm'] [usr]
        l_predict = [m for m , r in l_rank if m not in s_train]
        return l_predict[:int(self.args['top_k_item'])]
