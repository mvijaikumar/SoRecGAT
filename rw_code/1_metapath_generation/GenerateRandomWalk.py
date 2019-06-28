from Utilities import Utilities
from time import time
import numpy as np
import scipy.sparse as sp
import argparse,random
import pathlib2 as pathlib
from time import time
import sys

class GenerateRandomWalk:    
    def __init__(self,filepath,trust):
        self.trust = trust
        self.util = Utilities()
        t1 = time()
        print(filepath)
        self.user_item_dict,self.item_user_dict = self.util.read_matrix_as_dict(filepath)
        print('time taken {:.4}s'.format(time()-t1))

        self.user_lst   = list(set(self.user_item_dict.keys()))
        self.item_lst   = list(set(self.item_user_dict.keys()))
        #print(self.user_item_dict[0],self.user_item_dict[1])
        #print self.user_item_dict
        
    def generate_walks(self, outfilename, numwalks, walklength,metapath='UI',min_len_to_keep=5):
        print("generate walks")
        outfile  = open(outfilename, 'w')
        len_path = len(metapath)
        ind = 0 
        t1 = time()
        #for entity in self.item_lst:
        for entity in self.user_lst:##
            if ind%100 == 0: 
                print("entities completed: {} in {:.4}s".format(ind,time()-t1))
            ind += 1
            for j in range(numwalks): #wnum walks
                first_flag = True
                inst = entity
                outline = metapath[0] + str(inst)
                prv     = 0
                for i in range(1, walklength):
                    cur      = i%len_path
                    inst_lst = self.get_instances_for_metaclass(metapath[prv],metapath[cur],inst,first_flag)

                    first_flag = False
                    if inst_lst == None:
                        #print "empty list: ",metapath[prv],metapath[cur],inst
                        break
                    prv      = cur
                    inst     = inst_lst[0]
                    outline += " " + inst_lst[1]
                
                if (len(set(outline.split(" "))) > min_len_to_keep): #newly added this handles loop to some extent
                    outfile.write(outline + "\n")
        outfile.close()
    
    def get_instances_for_metaclass(self, prv, cur,inst,first_flag):
        #print("metaclass fun: ",prv,cur,inst)
        if prv == 'I' and cur == 'U':
            if inst not in self.item_user_dict:
                return None
            else: 
                user_lst = self.item_user_dict[inst]
            numu       = len(user_lst)
            userid   = random.randrange(numu)
            return (user_lst[userid],'U' + str(user_lst[userid]))
        elif prv =='U' and cur =='I':
            if inst not in self.user_item_dict:#viewed_item
                return None
            else:
                item_lst   = self.user_item_dict[inst]
            numi       = len(item_lst)
            itemid     = random.randrange(numi)
            if self.trust == True:
                return (item_lst[itemid],'U' + str(item_lst[itemid]))
            else:
                return (item_lst[itemid],'I' + str(item_lst[itemid]))
