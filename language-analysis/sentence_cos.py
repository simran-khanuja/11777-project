#%%
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import matplotlib.pyplot as plt
import json
import sentence_transformers.util as util
import torch.nn.functional as F
import random
import torch
import os
import amrlib
import penman
#os.chdir('language-analysis')
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
# sanity check, do the caption pairs always tend to be more similar than the negative samples
sanity_assertion_failed = 0
indices = []
for index, (cap, cap2) in enumerate(zip(caption_0s, caption_1s)):
    max_sim = F.cosine_similarity(c1,c2,dim=0)
    for index_2, cap_neg in enumerate(caption_1s):
        if F.cosine_similarity(c1,cap_neg,dim=0) > max_sim:
            indices.append((index, index_2))
            sanity_assertion_failed+=1
            break

print(sanity_assertion_failed, indices)
# prints nothing, => samples are fairly diverse 
# %%
# amr tree reconstruction
import nltk
from tqdm import tqdm

gtos = amrlib.load_gtos_model(device='cuda:0')
caption_amrs_0 = []
caption_amrs_1 = []
for caption in tqdm(caption_info_serialize):
    caption_amrs = caption['amr']
    caption_amrs_0.append(caption_amrs[0])
    caption_amrs_1.append(caption_amrs[1])

caption_0_hats, iscut = gtos.generate(caption_amrs_0)
caption_1_hats, iscut = gtos.generate(caption_amrs_1)
#%%
print(caption_0_hats)
#%%
def load_model():
    """Load the AMR parsing model from amrlib."""
    return amrlib.load_stog_model()


def get_amr_tree(amr_graph):
    """Generate an AMR tree for a given amrstring."""
    return penman.decode(amr_graph)


def compare_amr_trees(tree1, tree2):
    """Compare two AMR trees and return differences in nodes and relations."""
    nodes1 = {node for _, node, _ in tree1.triples}
    nodes2 = {node for _, node, _ in tree2.triples}
    relations1 = {(s, p, o) for s, p, o in tree1.triples}
    relations2 = {(s, p, o) for s, p, o in tree2.triples}

    common_nodes = nodes1.intersection(nodes2)
    unique_to_tree1_nodes = nodes1 - nodes2
    unique_to_tree2_nodes = nodes2 - nodes1

    common_relations = relations1.intersection(relations2)
    unique_to_tree1_relations = relations1 - relations2
    unique_to_tree2_relations = relations2 - relations1

    symmetric_diff = relations1.symmetric_difference(relations2)
    len_symmetric_diff = len(symmetric_diff)
    return {
        'common_nodes': common_nodes,
        'unique_to_tree1_nodes': unique_to_tree1_nodes,
        'unique_to_tree2_nodes': unique_to_tree2_nodes,
        'common_relations': common_relations,
        'unique_to_tree1_relations': unique_to_tree1_relations,
        'unique_to_tree2_relations': unique_to_tree2_relations,
        'number_of_symmetric_diffs': len_symmetric_diff
    }

def compare_nodes(amr1, amr2):
    nodes1 = set(amr1.nodes.keys())
    nodes2 = set(amr2.nodes.keys())
    common_nodes = nodes1.intersection(nodes2)
    unique_to_amr1 = nodes1 - nodes2
    unique_to_amr2 = nodes2 - nodes1
    return common_nodes, unique_to_amr1, unique_to_amr2


def compare_relations(amr1, amr2):
    relations1 = set(tuple(rel) for rel in amr1.relations)
    relations2 = set(tuple(rel) for rel in amr2.relations)
    common_relations = relations1.intersection(relations2)
    unique_to_amr1 = relations1 - relations2
    unique_to_amr2 = relations2 - relations1
    return common_relations, unique_to_amr1, unique_to_amr2

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

# with open('tree_dump.json', 'r') as infile:
#     caption_info_serialize = json.load(infile)
def merge_punct(doc):
    spans = []
    for word in doc[:-1]:
        if word.is_punct or not word.nbor(1).is_punct:
            continue
        start = word.i
        end = word.i + 1
        while end < len(doc) and doc[end].is_punct:
            end += 1
        span = doc[start:end]
        spans.append((span, word.tag_, word.lemma_, word.ent_type_))
    with doc.retokenize() as retokenizer:
        for span, tag, lemma, ent_type in spans:
            attrs = {"tag": tag, "lemma": lemma, "ent_type": ent_type}
            retokenizer.merge(span, attrs=attrs)
    return doc

def get_deps_diff(s1, s2, nlp):

    doc0 = nlp(s1)
    doc1 = nlp(s2)

    doc0 = merge_punct(doc0)
    doc1 = merge_punct(doc1)

    dep0 = set({token0.text: token0.dep_ for token0 in doc0}.items())
    dep1 = set({token1.text: token1.dep_ for token1 in doc1}.items())

    dep0_common_dep1 = dep0.intersection(dep1)
    dep0_notin_dep1 = dep0 - dep1
    dep1_notin_dep0 = dep1 - dep0
    diff_dep = dep0.symmetric_difference(dep1)
    len_sym_diff = len(diff_dep)
    
    return {
            'common_relations': dep0_common_dep1,
            'unique_to_tree1_relations': dep0_notin_dep1,
            'unique_to_tree2_relations': dep1_notin_dep0,
            'number_of_symmetric_diffs': len_sym_diff
        }

from nltk.tree import Tree
def get_cons_diff(tree1, tree2):
    # Convert strings to NLTK Trees
    tree1 = Tree.fromstring(tree1)
    tree2 = Tree.fromstring(tree2)

    # Get all nodes in the trees
    nodes1 = set([subtree.label() for subtree in tree1.subtrees()])
    nodes2 = set([subtree.label() for subtree in tree2.subtrees()])

    # Find unique and common nodes
    unique_nodes_len = len(set(nodes1).symmetric_difference(set(nodes2)))
    common_nodes = list(set(nodes1).intersection(set(nodes2)))
    n1_notin_n2 = nodes1-nodes2
    n2_notin_n1 = nodes2-nodes1

    return {
            'common_nodes': common_nodes,
            'unique_to_tree1_nodes': n1_notin_n2,
            'unique_to_tree2_nodes': n2_notin_n1,
            'number_of_symmetric_diffs': unique_nodes_len
        }


#%%
METEOR_scores_0 = []
METEOR_scores_1 = []
METEOR_with_caption = {
    # TO SIMRAN: USE THIS
}
from nltk.translate import meteor_score
from nltk import word_tokenize
import spacy
from tqdm import tqdm
mscores_baseline = []
for caption in caption_info_serialize:
    caption0, caption1 = caption['caption_pair']
    mscores_baseline.append(meteor_score.meteor_score([word_tokenize(caption0)],word_tokenize(caption1)))
print(sum(mscores_baseline)/len(mscores_baseline))


#%%

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('merge_noun_chunks')
# no pipeline to merge any punctuations, so used the code from the displacy visualizer (merge_punct)
# https://stackoverflow.com/questions/65083559/how-to-write-code-to-merge-punctuations-and-phrases-using-spacy
stog = load_model()


for ind, (cos, caption, caption_0_hat, caption_1_hat) in tqdm(enumerate(zip(cosine_distance_paired, caption_info_serialize, caption_0_hats, caption_1_hats))):
    caption_pair = caption['caption_pair']
    
    caption_pair_amr = caption['amr']
    cons_tree_pair = caption['constituency']

    METEORscore0 = meteor_score.meteor_score([word_tokenize(caption_pair[0])],word_tokenize(caption_0_hat))
    METEORscore1 = meteor_score.meteor_score([word_tokenize(caption_pair[1])],word_tokenize(caption_1_hat)) #nltk.translate.bleu_score.sentence_bleu([caption_pair[1].split()], caption_1_hat.split())
    METEOR_scores_0.append(METEORscore0)
    METEOR_scores_1.append(METEORscore1)

    deps_diff = get_deps_diff(caption_pair[0],caption_pair[1], nlp)
    cons_diff = get_cons_diff(cons_tree_pair[0], cons_tree_pair[1])
    amr_diffs = compare_amr_trees(get_amr_tree(caption_pair_amr[0]), get_amr_tree(caption_pair_amr[1]))

    METEOR_with_caption[caption['id']*2] = {
        'METEOR_score': METEORscore0, 
        'original': caption_pair[0], 
        'amr_reconstruct': caption_0_hat, 
        'cos_score_with_counterpart': float(cos), 
        'cons_diffs': cons_diff,
        'dependency_diffs': deps_diff,
        'amr_diffs': amr_diffs}

    METEOR_with_caption[caption['id']*2+1] = {
        'METEOR_score': METEORscore1, 
        'original': caption_pair[1], 
        'amr_reconstruct': caption_1_hat, 
        'cos_score_with_counterpart': float(cos), 
        'cons_diffs': cons_diff,
        'dependency_diffs': deps_diff,
        'amr_diffs': amr_diffs}

#%%
import json
with open('sentence_METEOR_amr_reconstruction.json','w') as f:
    f.write(json.dumps(METEOR_with_caption,indent=4,default=set_default))  

# %%
av0=sum(METEOR_scores_0)/len(METEOR_scores_0)
av1=sum(METEOR_scores_1)/len(METEOR_scores_1)
print((av0+av1)/2)
# %%
import json
with open('sentence_METEOR_amr_reconstruction.json','r') as f:
    sent_info = json.load(f)

sent_info = {k: v for k,v in sorted(sent_info.items(),key=lambda x: x[1]['METEOR_score'])} 

# %%
sent_info['381']
#%%
i = 0
for k,v in sent_info.items():
    print(v['original'],"\t", v['amr_reconstruct'], "\t", f"{v['METEOR_score']:.3f}")
    i+=1
    if i == 10:
        break
# %%
