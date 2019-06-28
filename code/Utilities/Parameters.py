import sys
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

class Parameters(object):
    def __init__(self,args,dataset):  
        # method =========================================================
        self.method               = args.method.lower()
        self.args                 = args
        self.result_path          = args.res_path + args.dataset + '/' + args.method + '/'
        self.item_embed_flag      = dataset.item_embed_flag 
        self.loss                 = args.loss
        
        # GAT ===========================================================
        if self.method in ['sorecgatitem','sorecgatuser']:
            self.attn_keep = args.attn_keep
            self.ffd_keep  = args.ffd_keep
            self.proj_keep = args.proj_keep
            self.hid_units = eval(args.hid_units)
            self.n_heads   = eval(args.n_heads)


        # count ===========================================================
        self.num_users            = dataset.num_user
        self.num_items            = dataset.num_item
        self.num_doms             = dataset.num_dom
        self.num_train_instances  = dataset.train_matrix.nnz ##len(dataset.trainArrQuadruplets[0])
        self.num_valid_instances  = len(dataset.validNegativesDict.keys())
        self.num_test_instances   = len(dataset.testNegativesDict.keys())
        # new
        self.tar_dom              = dataset.tar_dom 
         
        # data-structures ==================================================
        self.dom_item_dict        = dataset.dom_item_dict
        self.domain_matrix        = dataset.domain_matrix
        self.train_matrix         = dataset.train_matrix
        self.testNegativesDict    = dataset.testNegativesDict   
        self.validNegativesDict   = dataset.validNegativesDict

        # item_attr_related =============================================
        if self.item_embed_flag  == True:
            self.attr_dim         = dataset.attr_dim 
            self.attr_mat         = dataset.attr_mat
        
        self.dom_num_item        = dict()
        for ind in xrange(self.num_doms):
            self.dom_num_item[ind] = len(self.dom_item_dict[ind])
        
        # sorecgat    ============================
        if self.method.lower() in ['sorecgatitem','sorecgatuser']:
            self.adjacency_matrix    = dataset.adjacency_mat
            self.user_item_embed_mat = dataset.user_item_embed_mat
            self.social_embed_mat    = dataset.social_embed_mat
            self.num_nodes           = dataset.num_nodes
            self.feature_dim         = dataset.feature_dim
            self.social_attr_dim     = dataset.social_attr_dim

        # algo-parameters =======================================================
        self.num_epochs      = args.epochs
        self.batch_size      = args.batch_size
        self.valid_batch_siz = args.valid_batch_siz
        self.learn_rate      = args.lr
        self.optimizer       = args.optimizer
        
        # valid test =======================================================
        self.at_k        = args.at_k
        self.num_thread  = args.num_thread
        
        # hyper-parameters ======================================================
        self.num_factors   = args.num_factors
        self.num_negatives = args.num_negatives
        self.reg_w         = args.reg_Wh
        self.reg_b         = args.reg_bias
        self.reg_lam       = args.reg_lambda
        self.keep_prob     = args.keep_prob

        #=========================================================================
        # initializations
        if args.initializer == 'xavier':
            print('Initializer: xavier')
            self.initializer = tf.contrib.layers.xavier_initializer()
        elif args.initializer == 'random_normal':
            print('Initializer: random_normal')
            _stddev = args.stddev
            self.initializer = tf.random_normal_initializer(stddev=_stddev)
        elif args.initializer == 'random_uniform':
            print('Initializer: random_uniform')
            _min,_max = -args.stddev, args.stddev
            self.initializer = tf.random_uniform_initializer(minval=_min,maxval=_max)
            
    def get_adjacency_matrix_single(self,mat):
        return mat.todense().astype(np.int8)
