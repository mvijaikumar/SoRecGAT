import numpy as np
import subprocess
import argparse
import operator
import os
import argparse
import sys

def parse_args():
    # dataset and method
    parser = argparse.ArgumentParser(description='''Arguments; Eg: ./metapath2vec -train ../in_dbis/dbis.cac.w1000.l100.txt -output ../out_dbis/dbis.cac.w1000.l10
                                                    0 -pp 1 -size 128 -window 7 -negative 5 -threads 32''')
    # file names
    parser.add_argument('--exe_path', nargs='?',default='./.',help   ='path to executables.')    
    parser.add_argument('--dir', nargs='?', default='/path/',help='Use text data from <file> to train the model.')
    parser.add_argument('--dataset', nargs='?',      default='music',help='(music or filmtrust.social) dataset name (this name is used for concatenated files).')
    parser.add_argument('--metapath_list',nargs='?', default='.US', help='metapath lists with space dilimeter.') #default='.ID .IV .IB'
    #===============

    parser.add_argument('--size', type=int, default=64,      help='Set size of word vectors; default is 100.')
    parser.add_argument('--save_vocab', nargs='?', default='',help='The vocabulary will be saved to <file>.')
    parser.add_argument('--read_vocab', nargs='?', default='',help='''The vocabulary will be read from <file>,not constructed from the training data.''')
    parser.add_argument('--debug', type=int, default=2,       help='Set the debug mode (default = 2 = more info during training).')
    parser.add_argument('--alpha', type=float, default=0.025, help='Set the starting learning rate; default is 0.025 for skip-gram.')
    parser.add_argument('--window', type=int, default=5,      help='Set max skip length between words; default is 5(mine 7).')
    parser.add_argument('--sample', type=float, default=1e-3, help='''Set threshold for occurrence of words. 
                        Those that appear with higher frequency in the training data. will be randomly down-sampled; default is 1e-3,useful range is (0, 1e-5)''')
    parser.add_argument('--pp', type=int, default=0,  help='''Use metapath2vec++ or metapath2vec; default is 0  (metapath2vec); for metapath2vec++, use 1.''')
    parser.add_argument('--negative', type=int, default=5,    help='''Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)''')
    parser.add_argument('--threads', type=int, default=12,    help='Use <int> threads (default 12).')
    parser.add_argument('--iter', type=int, default=15,        help='Run more training iterations (default 5).')
    parser.add_argument('--min_count', type=int, default=10,   help='''This will discard words that appear less than <int> times; default is 5.''')
    parser.add_argument('--classes', type=int, default=0,
            help='''Output word classes rather than word vectors; default number of classes is 0 (vectors are written)''')
    return parser.parse_args()

class EmbeddingGeneration:
    def __init__(self,args):
        self.args = args
        self.get_concat_files(direc=self.args.dir+self.args.dataset+'.walks',finame_lst=self.args.metapath_list.split(" "),foutname=self.args.dataset+'.walks')
        self.generate_embeddings(self.args)
        self.store_embeddings(self.args.dir+self.args.output+'.embed.txt',self.args.dir + self.args.output)
        
    def store_embeddings(self, fin_name, fout_name):
        print(fin_name,fout_name)
        with open(fin_name) as fin, open(fout_name+'.social_embed.deepwalk','w') as fsocial, open(fout_name+'.entity_embed.deepwalk','w') as fentity:
            for line in fin:
                line = line.strip()
                wrds = line.split(' ')
                if 'S' == wrds[0][0]:
                    fsocial.write(wrds[0].replace('S','')+'::'+' '.join(wrds[1:])+'\n')
                if 'U' == wrds[0][0]:
                    fentity.write(wrds[0].replace('U','')+'::'+' '.join(wrds[1:])+'\n')

    def get_concat_files(self,direc, finame_lst, foutname):
        print("finname_lst", finame_lst,foutname)
        foutname = direc #+ foutname
        ftemp    = direc + 'temp'
        fout     = open(foutname,'w')
        fout.close()

        for finame in finame_lst:
            finame  = direc + finame
            print(finame)

            catfile = open(ftemp,'w')
            subprocess.call(['cat',finame,foutname],stdout=catfile)
            catfile.close()
            subprocess.call(['mv',ftemp,foutname])    
        subprocess.call(['shuf', foutname, '-o',foutname])
    
    def generate_embeddings(self,args):
        subprocess.call(['make','-I',args.exe_path+'/.','clean'])
        subprocess.call(['make','-I',args.exe_path+'/.'])
        print('Before calling main metapath2vec')
        print(args)
        subprocess.call(['./metapath2vec',#args.exe_path + '/metapath2vec',
                     '-train',    args.dir + args.dataset+'.walks',
                     '-output',   args.dir + args.output+'.embed',
                     '-size',     str(args.size),
                     '-window',   str(args.window),
                     '-sample',   str(args.sample),
                     '-pp',       str(args.pp),
                     '-negative', str(args.negative),
                     '-threads',  str(args.threads),
                     '-iter',     str(args.iter),
                     '-min-count',str(args.min_count),
                     '-alpha',    str(args.alpha),
                     '-classes',  str(args.classes),
                     '-debug',    str(args.debug), 
                     '-save-vocab',  str(args.save_vocab), 
                     '-read-vocab',  str(args.read_vocab)]) 
        print('After calling main metapath2vec')
        
    def load_ids(self,fname):
        item_id = dict()
        for line in open(fname):
            toks = line.split("\t")
            item,idval = toks[0], int(toks[1]) 
            item_id[idval] = item
        return item_id
    
    def convert_to_old_ids(self,finname,foutname):
        print(finname,foutname)
        fin  = open(finname,'r')
        fout = open(foutname,'w')
        
        for line in fin:        
            line = line.replace("\n","")
            toks = line.split(" ")
            if len(toks) <= 2 :
                continue
            item  = int(toks[0][1:]) ## to remove item prefix
            embed = line.replace(toks[0]+" ","")
            fout.write(self.item_id_dict[item] + "::" + embed + "\n")
                
        fin.close()
        fout.close()
    
if __name__ == '__main__':
    args = parse_args()
    args.output            = args.dataset
    args.item_embed_output = args.dataset
    args.embed_item_id     = args.dataset
    
    emb = EmbeddingGeneration(args)
    filenames = ['.embed.txt','.embed','.walks','.entity_embed.deepwalk']
    for fil in filenames:
        subprocess.call(['rm',args.dir + args.dataset + fil])    
    subprocess.call(['mv',args.dir + args.dataset  + '.social_embed.deepwalk',args.dir + args.dataset.replace('.social','') + '.social_embed.final'])    
    print("Finished...")
    
