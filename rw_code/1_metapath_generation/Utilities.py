import numpy as np
import scipy.sparse as sp
import sys
import math
import argparse
from time import time
import pathlib2 as pathlib
from collections import defaultdict
import random

class Utilities:
    def __init__(self):
        pass
    def get_counts(self,filename):
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        return (num_users+1,num_items+1)
    
    # only for 2 column files
    def read_matrix_as_dict(self,fname):
        print('   Inside read matrix as dict fname: ' + fname)
        item_entity = dict()
        entity_item = dict()
        with open(fname,'r') as fin:
            for line in fin:
                toks = line.strip().split("\t")
                if len(toks) >= 2:
                    item, entity = int(toks[0]), int(toks[1])
                    if item not in item_entity:
                        item_entity[item] = []
                    if entity not in item_entity[item]:# to avoid duplicates
                        item_entity[item].append(entity)
                    if entity not in entity_item:
                        entity_item[entity] = []
                    if item not in entity_item[entity]:# to avoid duplicates
                        entity_item[entity].append(item)
        return item_entity, entity_item
    
    def read_matrix_as_sparse(self,fname,dim):
        print('   Inside read sparse matrix fname: ' + fname)
        mat = sp.lil_matrix(dim, dtype=np.float32)
        with open(fname, "r") as fin:
            line = fin.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = rating
                line = fin.readline()    
        return mat
    
    def generate_trantition_matrix_from_sparse(self,sparse_mat):
        ##sparse_mat = np.ca
        num_rows,num_cols = sparse_mat.shape
        transition_mat = sp.csc_matrix((num_rows, num_cols), dtype=np.float32)

        col_sum_vector = 1./np.array(sparse_mat.sum(axis=0))
        col_sum_vector[np.isinf(col_sum_vector)] = 0
        col_sum_vector = sp.lil_matrix(col_sum_vector.flatten())
        # diagonalize the vector and multiply
        transition_mat = sparse_mat.multiply(col_sum_vector) ## check broadcasting once more
        return transition_mat
    
    def generate_dict(self, sparse_mat):
        print("Generating dictionary...")
        sparse_mat = sparse_mat.tolil()
        siz        = sparse_mat.get_shape()
        mat_dict   = dict()

        rowset = set(sparse_mat.tocoo().row)
        for node in rowset:
            dist_dict = dict()
            vec_col = sparse_mat[node].tocoo().col
            vec_dat = sparse_mat[node].tocoo().data
            for ind in range(len(vec_col)):
                dist_dict[int(vec_col[ind])] = float(vec_dat[ind])            
            mat_dict[int(node)] = dist_dict
        return mat_dict
        
    def get_random_id(self,dist_vect):
        VA  = VoseAlias(dist_vect)
        res = VA.sample_n(size=1) #one random index
        return res[0] 
