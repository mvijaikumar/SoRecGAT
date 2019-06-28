import numpy as np
import scipy.sparse as sp
from time import time
from Gat_Dataset import Gat_Dataset

import subprocess as sub
import pandas as pd
import sys,pdb

class SocialItem_Dataset(Gat_Dataset):
    def __init__(self,args):
        Gat_Dataset.__init__(self,args)
        
        self.social_embed_path  = self.get_embed_path(self.path,args.dataset)
	self.num_social         = self.get_num_social(self.social_embed_path + '.social')

        self.social_attr_dim    = self.get_item_embed_dim(self.social_embed_path + '.socialitem_embed.final')
        self.social_embed_mat   = self.load_social_embed_as_mat(self.social_embed_path + '.socialitem_embed.final')

        # adjacency matrix related
	self.socialitem_adjacency_mat = self.load_social_file_as_matrix(self.social_embed_path + '.social')
        self.adjacency_mat            = self.get_socialitem_adjacency_matrix_sparse(mat1=self.adjacency_mat.tolil(), mat2=self.socialitem_adjacency_mat.tolil())
        self.num_nodes                = self.adjacency_mat.shape[0]
        print("num nodes: ",self.num_nodes)
     
    def get_num_social(self,fname):
        df = pd.read_csv(fname,delimiter='\t',header=None)
        return df[1].max()+1

    def load_social_file_as_matrix(self, filename):        
        # Construct matrix
        mat     = sp.dok_matrix((self.num_item,self.num_social), dtype=np.int32) ##item
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                if len(arr) > 2:
                    item, social, rating = (int(arr[0]), int(arr[1]), int(arr[2]))
                else:
                    item, social, rating = (int(arr[0]), int(arr[1]),1)
                if (rating > 0) and item < self.num_item: ##for item
                    mat[item, social] = 1
                line = f.readline()    
        return mat

    def load_social_embed_as_mat(self, filename):
        # Construct matrix
        mat = np.zeros((self.num_social,self.social_attr_dim),dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                toks    = line.replace("\n","").split("::")
                itemid  = int(toks[0])
                embed   = np.array(toks[1].split(" ")).astype(np.float)
                mat[itemid] = embed
                line = f.readline()
        return mat

    def get_socialitem_adjacency_matrix_sparse(self,mat1,mat2): ## exactly same as param
        num_row,num_col = (mat1.shape[0] + mat2.shape[1], mat1.shape[0] + mat2.shape[1])
        mat = sp.lil_matrix((num_row,num_col),dtype=np.int8)

        assert num_row == num_col, 'In adj matrix conv. row and col should be equal.'
        mat[0:mat1.shape[0],0:mat1.shape[0]]             = mat1.astype(np.int8)
        mat[self.num_user:self.num_user+self.num_item, mat1.shape[0]:] = mat2.astype(np.int8)
        mat[mat1.shape[0]:,self.num_user:mat1.shape[0]]  = mat2.astype(np.int8).T

        return mat.tocsr()

