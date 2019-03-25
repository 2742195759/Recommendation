
        with tf.device('/gpu:1') :
            print '[INFO] Start Model Build'
            R_uu = self.R_uu = tf.placeholder('float64' , self.args['a_uu'].shape)
            R_um = self.R_um = tf.placeholder('float64' , self.args['a_um'].shape)
            R_uAm = self.R_uAm = tf.placeholder('float64' , self.args['a_uAm'].shape)
            R_uAm = tf.sqrt(R_uAm)

            R_similar = R_uu / R_uAm / tf.reshape(R_uAm , (1,-1))

            R_rank = tf.nn.top_k(R_similar , int(self.args['top_k'])).indices
            R_mask = tf.cast(xktf.multi_hots(R_rank , R_similar.shape[0] , device='/cpu:0') , 'float64')
            R_umrate = tf.matmul(tf.multiply(R_similar , R_mask ), R_um)
            R_ans = tf.nn.top_k(R_umrate , int(self.args['top_k_item'])).indices

            self.R_ans = R_ans  
            print '[INFO] Start Model Run'
            data = self.args
            feed_data = {self.R_uu : data['a_uu'] , self.R_um : data['a_um'] , self.R_uAm : data['a_uAm']}
            with tf.Session() as ss : 
                #pdb.set_trace()
                #print ss.run(R_um , feed_data )
                #print ss.run(R_umrate , feed_data )
                a_u_rankm = ss.run(self.R_ans , feed_data )
            print a_u_rankm
            self.d_u_spredictm = {}
            for t_u , t_lm in enumerate(a_u_rankm) : 
                self.d_u_spredictm[t_u] = set(t_lm)
