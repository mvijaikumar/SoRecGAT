# SoRecGAT: Leveraging Graph Attention Mechanism for Top-N Social Recommendation

This is our tensorflow implementation of the paper "SoRecGAT: Leveraging Graph Attention Mechanism for Top-N Social Recommendation SoRecGAT implementation".

## Environment

tensorflow-gpu == 1.12.0

numpy == 1.16.0

Python == 2.7.12

scipy == 1.2.0

## Example to Run the Codes

### For Music dataset
> python Main.py --method sorecgatitem --path ../data/music/ --dataset music --res_path ../results/ --epochs 60 --batch_size 1024 --valid_batch_siz 256 --lr 0.0004 --initializer xavier --stddev 0.02 --optimizer rmsprop --loss ce --num_factors 64 --num_negatives 1 --keep_prob 0.5 --attn_keep 1.0 --ffd_keep 1.0 --proj_keep 1.0 --hid_units [32] --n_heads [12,6] --at_k 5 --num_thread 8

### For Art dataset
> python Main.py --method sorecgatuser --path ../data/art/ --dataset art --res_path ../results/ --epochs 60 --batch_size 1024 --valid_batch_siz 256 --lr 0.0001 --initializer xavier --stddev 0.02 --optimizer rmsprop --loss ce --num_factors 64 --num_negatives 1 --keep_prob 0.5 --attn_keep 1.0 --ffd_keep 1.0 --proj_keep 0.7 --hid_units [32] --n_heads [12,6] --at_k 5 --num_thread 8

## Dataset Description

### Music Dataset

- music.train
  - Train file
  - Each line contains UserID, ItemID, ImplicitRating and Some extra details (tab separated)

- music.valid
  - Valid file 
  - Each line contains (UserID, ItemID, extra detail)::negative item list

- music.test
  - Test file
  - Each line contains (UserID, ItemID, extra detail)::negative item list (space separated)

- music.social
  - Social interaction file
  - Each line contains ItemID, SocialEntityID (tab separated)

- music.user_embed.final
  - User embedding file
  - Each line contains UserID::Initial User Embedding (space separated)

- music.item_embed.final
  - Item embedding file
  - Each line contains ItemID::Initial Item Embedding (space separated)

- music.socialitem_embed.final
  - Social entity embedding file
  - Each line contains SocialEntityID::Initial SocialEntity Embedding (space separated)

### Art Dataset

- art.train
  - Train file
  - Each line contains UserID, ItemID, ImplicitRating and some extra details (tab separated)

- art.valid
  - Valid file 
  - Each line contains (UserID, ItemID, extra details)::negative item list (space separated)

- art.test
  - Test file
  - Each line contains (UserID, ItemID, extra details)::negative item list (space separated)

- art.social
  - Social interaction file
  - Each line contains UserID, SocialEntityID (tab separated)

- art.user_embed.final
  - User embedding file
  - Each line contains UserID::Initial User Embedding (space separated)

- art.item_embed.final
  - Item embedding file
  - Each line contains ItemID::Initial Item Embedding (space separated)

- art.socialitem_embed.final
  - Social entity embedding file
  - Each line contains SocialEntityID::Initial SocialEntity Embedding (space separated)
  
Link for all the datasets: https://www.dropbox.com/sh/3bkratvwuhgzctw/AABIA2GxPy4KZbmX0Do4S8b5a?dl=0
