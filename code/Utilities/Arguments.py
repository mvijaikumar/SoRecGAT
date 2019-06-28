import argparse
import sys
def parse_args():
    # dataset and method
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--method', nargs='?', default='sorecgatitem',help='sorecgatuser,sorecgatitem')
    parser.add_argument('--path', nargs='?',
                        default='../data/music/',
                        help   ='Input data path.')
    parser.add_argument('--dataset', nargs='?', 
                        default='music',
                        #default='art',
                        help='Choose a dataset.')
    parser.add_argument('--res_path', nargs='?',default='../result/',help='result path for plots and best error values.')
    parser.add_argument('--res_folder', nargs='?',default='',help='specific folder corresponding to different runs on different parameters.')

    # algo-parameters
    parser.add_argument('--epochs', type=int, default=1,help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,help='Batch size.')
    parser.add_argument('--valid_batch_siz', type=int, default=256,help='Valid batch size.')
    parser.add_argument('--lr', type=float, default=.0001,help='Learning rate.')
    parser.add_argument('--initializer', nargs='?', default='xavier',help='random_normal,random_uniform,xavier')
    parser.add_argument('--stddev', type=float, default=0.02,help='stddev for normal and [min,max] for uniform')
    parser.add_argument('--optimizer', nargs='?', default='rmsprop',help='rmsprop, adam')
    # new
    parser.add_argument('--loss', nargs='?', default='ce',help='ce')
    
    # hyper-parameters
    parser.add_argument('--num_factors', type=int, default=64,  help='Embedding size.')
    parser.add_argument('--num_layers', type=int, default=2,    help='Number of hidden layers.') # feature in testing ##not completed
    parser.add_argument('--num_negatives', type=int, default=1, help='Negative instances in sampling.')
    parser.add_argument('--reg_Wh', type=float, default=0.0000, help="Regularization for weight vector.")
    parser.add_argument('--reg_bias', type=float, default=0.000,help="Regularization for user and item bias embeddings.")
    parser.add_argument('--reg_lambda', type=float, default=0.000,help="Regularization lambda for user and item embeddings.")
    parser.add_argument('--keep_prob', type=float, default=0.5, help='droupout keep probability in layers.')

    # graph
    parser.add_argument('--attn_keep', type=float, default=1.0, help='attn keep probability in layers.')
    parser.add_argument('--ffd_keep' , type=float, default=1.0, help='ffd keep probability in layers.')
    parser.add_argument('--proj_keep', type=float, default=1.0, help='proj keep probability in projection weights layers for reviews.')
    parser.add_argument('--hid_units', nargs='?', default='[32]',help='hidden units of GAT')
    parser.add_argument('--n_heads',   nargs='?', default='[12,6]',help='number of heads of GAT')

    # valid and test
    parser.add_argument('--at_k', type=int, default=5,help='@k for hit ratio, ndcg, etc.')
    parser.add_argument('--num_thread', type=int, default=10,help='number of threads.')
    parser.add_argument('--comment', nargs='?', default='comment',help='comments about the current experimental iterations.')
    
    return parser.parse_args()       
