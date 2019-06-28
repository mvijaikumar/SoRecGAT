import tensorflow as tf
import numpy as np
import sys,pdb
sys.path.append('./.')
sys.path.append('./utils/.')
sys.path.append('./utils')
sys.path.append('../code/utils/.')

########################################################################
# The following packages (models and utils) are adapted from Peter/GAT #
########################################################################
from models import SpGAT
from utils import process

class SoRecGAT(object):
    def __init__(self,params):
        
        self.num_factors      = params.num_factors        
        self.num_users        = params.num_users
        self.num_items        = params.num_items  
        self.num_doms         = params.num_doms
        self.reg_lam          = params.reg_lam
        self.reg_w            = params.reg_w
        self.reg_b            = params.reg_b
        self.initializer      = params.initializer        
        self.params           = params

        self.num_nodes        = params.num_nodes
        self.feature_dim      = params.feature_dim
        self.social_attr_dim  = params.social_attr_dim
        print('feature dim, social_attr_dim: ', self.feature_dim,self.social_attr_dim)
        self.model            = SpGAT()
        self.biases           = process.preprocess_adj_bias(params.adjacency_matrix)

        self.user_item_features   = params.user_item_embed_mat#[np.newaxis]
        self.social_features      = params.social_embed_mat#[np.newaxis]

        self.out_size  = params.num_factors 
	self.hid_units = params.hid_units 
        self.n_heads   = params.n_heads 
        self.attn_keep = params.attn_keep 
        self.ffd_keep  = params.ffd_keep 
        self.proj_keep = params.proj_keep 

        self.residual     = False
        self.nonlinearity = tf.nn.elu

    def define_model(self,user_indices,item_indices,dom_indices,true_rating,keep_prob,batch_siz,valid_clip):

        self.user_indices         = user_indices
        self.item_indices         = item_indices
        self.dom_indices          = dom_indices
        self.true_rating          = true_rating
        self.keep_prob            = keep_prob  
        self.valid_clip           = valid_clip  
        self.is_training          = tf.equal(0.0,valid_clip)
        
        self.X_features_user_item = tf.Variable(self.user_item_features,trainable=False,dtype=tf.float32,name='X_features_user_item')
        self.X_features_social    = tf.Variable(self.social_features,trainable=False,dtype=tf.float32,name='X_features_social')

        self.weight_proj_social   = tf.Variable(self.initializer(shape=[self.social_attr_dim,self.feature_dim]),dtype=tf.float32,name='weight_proj')
        self.b_proj_social        = tf.Variable(self.initializer(shape=[1,self.feature_dim]),dtype=tf.float32,name='weight_proj')
        self.X_features_social    = tf.matmul(self.X_features_social,self.weight_proj_social) + self.b_proj_social
        self.X_features_social    = tf.nn.dropout(self.X_features_social, self.valid_clip + (1-self.valid_clip) * self.proj_keep)

        self.X_features           = tf.expand_dims(tf.concat([self.X_features_user_item,self.X_features_social],axis=0),0)
        self.bias_in              = tf.SparseTensor(indices=self.biases[0],values=self.biases[1],dense_shape=self.biases[2])

        self.logits               = self.model.inference(self.X_features, self.out_size, nb_nodes=self.num_nodes, training=self.is_training,
                                    attn_drop = (1-self.attn_keep) * (1-self.valid_clip), ffd_drop = (1-self.ffd_keep) * (1-self.valid_clip),
                                    bias_mat=self.bias_in,
                                    hid_units=self.hid_units, n_heads=self.n_heads,
                                    residual=self.residual, activation=self.nonlinearity)

        self.user_item_embeddings = tf.reshape(self.logits, [-1, self.n_heads[-1] * self.out_size])         
        self.user_embeds          = tf.nn.embedding_lookup(self.user_item_embeddings, self.user_indices)
        self.item_embeds          = tf.nn.embedding_lookup(self.user_item_embeddings, self.num_users + self.item_indices)

        
        self.multiplied_output1   = tf.multiply(self.user_embeds,self.item_embeds)
        self.multiplied_output1   = tf.nn.dropout(self.multiplied_output1, self.keep_prob)
        
        self.mult_cat             = tf.concat([self.multiplied_output1],axis=1)
        self.w                    = tf.Variable(self.initializer(shape=[self.n_heads[-1] * self.out_size,1]),dtype=tf.float32,name='w')
        self.pred_rating          = tf.reshape((tf.nn.sigmoid(tf.matmul(self.mult_cat,self.w))),shape=[-1])

    def define_loss(self,loss_type='all'):
        #self.regularization_loss = tf.constant(0.0)
        self.regularization_loss = self.reg_w * (tf.nn.l2_loss(self.w) + tf.nn.l2_loss(self.weight_proj_social)) + self.reg_b * (tf.nn.l2_loss(self.b_proj_social))
