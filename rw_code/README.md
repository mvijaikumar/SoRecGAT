# Initial Embedding Generation

This directory contains code for constructing initial embeddings for user-item interaction network and social network.

- 1_metapath_generation directory contains code for generating random walks

- 2_embed_generation directory contains code for generating embeddings for nodes

# Example to Run the Codes

## 1_metapath_generation

### For user-item interaction matrix

> python Metapath_main.py --dir ../data/art/ --dataset art.train --numwalks 40 --walklength 100 --min_len_to_keep 5 --metapath UI

### For social network
 
> python Metapath_main_social.py --dir ../data/art/ --dataset art.social --numwalks 40 --walklength 100 --min_len_to_keep 5 --metapath US --two_sided 1

## 2_embed_generation

### For user-item interaction matrix

> python EmbeddingGeneration.py --dir ../data/art/ --dataset art.train --metapath_list .UI --size 64 --window 5 --iter 15 --min_count 10 --alpha 0.025
    
### For social network
 
> python EmbeddingGenerationSocial.py --dir ../data/art/ --dataset art.social --metapath_list .US --size 64 --window 5 --iter 15 --min_count 10 --alpha 0.025
