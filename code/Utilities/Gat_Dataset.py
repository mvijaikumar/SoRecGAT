import numpy as np
import scipy.sparse as sp
from time import time
from Dataset import Dataset
import subprocess as sub
import pandas as pd

class Gat_Dataset(Dataset):
    def __init__(self,args):
        Dataset.__init__(self,args)
        self.user_item_embed_mat   = self.get_user_item_embed_mat(self.user_attr_mat,self.attr_mat)
        #self.adjacency_mat         = self.get_adjacency_matrix(self.train_matrix,self.train_matrix.T)
        self.adjacency_mat         = self.get_adjacency_matrix_sparse(self.train_matrix,self.train_matrix.T)
        assert self.adjacency_mat.shape[0] == self.adjacency_mat.shape[1]
        self.num_nodes             = self.adjacency_mat.shape[0]
        self.feature_dim           = self.attr_dim
        print("num nodes: ",self.num_nodes)
    
    def get_user_item_embed_mat(self, mat1, mat2):
        return np.concatenate([mat1,mat2],axis=0)

    def get_adjacency_matrix(self,mat1,mat2): ## exactly same as param
        num_row,num_col = (mat1.shape[0] + mat2.shape[0], mat1.shape[1] + mat2.shape[1])
        mat = np.zeros([num_row,num_col],dtype=np.int8)
        assert num_row==num_col, 'In adj matrix conv. row and col should be equal.'
        mat[0:mat1.shape[0],mat1.shape[0]:] = mat1.todense().astype(np.int8)
        mat[mat1.shape[0]:,0:mat1.shape[0]] = mat2.todense().astype(np.int8)
        return sp.csr_matrix(mat)

    def get_adjacency_matrix_sparse(self,mat1,mat2): ## exactly same as param
        num_row,num_col = (mat1.shape[0] + mat2.shape[0], mat1.shape[1] + mat2.shape[1])
        mat = sp.lil_matrix((num_row,num_col),dtype=np.int8)
        assert num_row == num_col, 'In adj matrix conv. row and col should be equal.'
        mat[0:mat1.shape[0],mat1.shape[0]:] = mat1.astype(np.int8).tolil()
        mat[mat1.shape[0]:,0:mat1.shape[0]] = mat2.astype(np.int8).tolil()
        return mat.tocsr()
