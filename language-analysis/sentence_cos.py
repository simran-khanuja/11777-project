#%%
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import matplotlib.pyplot as plt
import json
import sentence_transformers.util as util
import torch.nn.functional as F
import random
import torch

# %% 
# Visualize sentence transformer embeddings
# load the caption tree dump, and extract the captions from it
caption_info_serialize = ""
with open('tree_dump.json','r') as f:
     caption_info_serialize = f.read()
     caption_info_serialize = json.loads(caption_info_serialize)

# %%
minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

caption_0s = []
caption_1s = []
for caption in caption_info_serialize:
    caption_pair = caption['caption_pair']
    embeddings = minilm.encode(caption_pair)
    caption_0s.append(torch.tensor(embeddings[0]))
    caption_1s.append(torch.tensor(embeddings[1]))
# %%
print(caption_0s[0].shape)
#%%
# get average cosine distances b/w paired captions, and unpaired captions
def get_negative_sample(c2, caption_1s):
    while True:
        new_c = random.sample(caption_1s,1)
        if not torch.equal(new_c[0],c2):
            return new_c[0]
        
cosine_distance_paired = []
cosine_distance_negative_sampled = []
for c1, c2 in zip(caption_0s, caption_1s):
     cosine_distance_paired.append(F.cosine_similarity(c1,c2,dim=0))
     cosine_distance_negative_sampled.append(F.cosine_similarity(c1,get_negative_sample(c2,caption_1s),dim=0))

print(f"Average cosine distance: {sum(cosine_distance_paired)/len(cosine_distance_paired)}")
print(f"Average negative sampled cosine distance: {sum(cosine_distance_negative_sampled)/len(cosine_distance_negative_sampled)}")

# %%
