import tensorflow as tf

'''
Build the graph of EFM
'''

class EFM:
    def __init__(self, args):
        self.args = args
        self.INFINITY = 10e+12

        self.user_id = tf.placeholder(tf.int32, [None])
        self.item_id = tf.placeholder(tf.int32, [None])
        self.feature_id = tf.placeholder(tf.int32, [None])
        self.a_ui = tf.placeholder(tf.float32, [None])
        self.x_uf = tf.placeholder(tf.float32, [None])
        self.y_if = tf.placeholder(tf.float32, [None])

        self.initializer = tf.random_uniform_initializer(minval=0, maxval=1)
        self.user_embedding_matrix = tf.get_variable("user_embedding_matrix", [self.args['user_number'], self.args['embed_dim_r']],
                                              initializer=self.initializer)
        self.item_embedding_matrix = tf.get_variable("item_embedding_matrix", [self.args['item_number'], self.args['embed_dim_r']],
                                              initializer=self.initializer)
        self.user_h_embedding_matrix = tf.get_variable("user_h_embedding_matrix", [self.args['user_number'], self.args['embed_dim_r_']],
                                              initializer=self.initializer)
        self.item_h_embedding_matrix = tf.get_variable("item_h_embedding_matrix", [self.args['item_number'], self.args['embed_dim_r_']],
                                              initializer=self.initializer)
        self.feature_embedding_matrix = tf.get_variable("feature_embedding_matrix", [self.args['feature_number'], self.args['embed_dim_r']],
                                              initializer=self.initializer)

        self.embedded_user = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_id)
        self.embedded_item = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_id)
        self.embedded_user_h = tf.nn.embedding_lookup(self.user_h_embedding_matrix, self.user_id)
        self.embedded_item_h = tf.nn.embedding_lookup(self.item_h_embedding_matrix, self.item_id)
        self.embedded_feature = tf.nn.embedding_lookup(self.feature_embedding_matrix, self.feature_id)
        self.complete_embedded_user = tf.concat([self.embedded_user, self.embedded_user_h], 1)
        self.complete_embedded_item = tf.concat([self.embedded_item, self.embedded_item_h], 1)


    def train_op(self):
        if self.args['learning_rate'] == 'sgd':
            self.train_op = tf.train.GradientDescentOptimizer(self.args['learning_rate']).minimize(self.loss)
        else:
            self.train_op = tf.train.AdagradOptimizer(self.args['learning_rate']).minimize(self.loss)

    def prediction(self):
        self.a_predict = tf.reduce_sum(tf.multiply(self.complete_embedded_user, self.complete_embedded_item),1)
        self.x_predict = tf.reduce_sum(tf.multiply(self.embedded_user, self.embedded_feature),1)
        self.y_predict = tf.reduce_sum(tf.multiply(self.embedded_item, self.embedded_feature),1)


    def loss(self):
        self.u_reg = tf.reduce_sum(tf.abs(self.user_embedding_matrix) - self.user_embedding_matrix)
        self.i_reg = tf.reduce_sum(tf.abs(self.item_embedding_matrix) - self.item_embedding_matrix)
        self.u_h_reg = tf.reduce_sum(tf.abs(self.user_h_embedding_matrix) - self.user_h_embedding_matrix)
        self.i_h_reg = tf.reduce_sum(tf.abs(self.item_h_embedding_matrix) - self.item_h_embedding_matrix)
        self.f_reg = tf.reduce_sum(tf.abs(self.feature_embedding_matrix) - self.feature_embedding_matrix)

        self.non_negative_reg = self.INFINITY * (self.u_reg + self.i_reg + self.u_h_reg + self.i_h_reg + self.f_reg)
        self.l2_reg = self.args['reg_u'] * (tf.nn.l2_loss(self.user_embedding_matrix) +
                                            tf.nn.l2_loss(self.item_embedding_matrix)) + \
                      self.args['reg_h'] * (tf.nn.l2_loss(self.user_h_embedding_matrix) +
                                            tf.nn.l2_loss(self.item_h_embedding_matrix) +
                                            tf.nn.l2_loss(self.feature_embedding_matrix))
        self.error = tf.reduce_sum(tf.pow(self.a_ui - tf.reduce_sum(tf.multiply(self.complete_embedded_user, self.complete_embedded_item),1), 2)+
                                   tf.pow(self.x_uf - tf.reduce_sum(tf.multiply(self.embedded_user, self.embedded_feature), 1), 2)+
                                   tf.pow(self.y_if - tf.reduce_sum(tf.multiply(self.embedded_item, self.embedded_feature), 1), 2))

        self.loss = self.error + self.non_negative_reg + self.l2_reg
