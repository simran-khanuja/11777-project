# %%
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from datasets import load_dataset
import stanza
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import amrlib
import spacy
import matplotlib.pyplot as plt
from   amrlib.graph_processing.amr_plot import AMRPlot
# %%
dataset = load_dataset('facebook/winoground',use_auth_token=True)
# %%
winoground_text = dataset['test'].select_columns(['caption_0','caption_1'])
# %%
conParser = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,constituency,depparse', use_gpu=True)
# %%

caption_info = defaultdict(dict)
nlp = spacy.load('en_core_web_sm')
stog = amrlib.load_stog_model(device='cuda:0')

amrlib.setup_spacy_extension()
doc1_sentences = doc2_sentences = []

for tup in tqdm(winoground_text):
    doc1 = conParser(tup['caption_0'])
    doc2 = conParser(tup['caption_1'])
    doc1_s = nlp(tup['caption_0'])
    doc2_s = nlp(tup['caption_1'])
    doc1_cons = [sentence.constituency for sentence in doc1.sentences] # use the read_tree method in here: https://github.com/stanfordnlp/stanza/blob/main/stanza/models/constituency/tree_reader.py to get it back
    doc2_cons = [sentence.constituency for sentence in doc2.sentences]

    doc1_deps = [{'id':word.id, 'word': word.text, 'head id': word.head, 'head': sent.words[word.head-1].text if word.head > 0 else "root", 'deprel': word.deprel} for sent in doc1.sentences for word in sent.words]
    doc2_deps = [{'id':word.id, 'word': word.text, 'head id': word.head, 'head': sent.words[word.head-1].text if word.head > 0 else "root", 'deprel': word.deprel} for sent in doc2.sentences for word in sent.words]

    doc1_graphs = stog.parse_sents([sent.text for sent in doc1_s.sents])
    doc2_graphs = stog.parse_sents([sent.text for sent in doc2_s.sents])

    caption_info[(tup['caption_0'],tup['caption_1'])]['constituency'] = [doc1_cons, doc2_cons]
    caption_info[(tup['caption_0'],tup['caption_1'])]['dependency'] = [doc1_deps, doc2_deps]
    caption_info[(tup['caption_0'],tup['caption_1'])]['amr'] = [doc1_graphs, doc2_graphs]

# %%
caption_info_serialize = []
i = 0
for k,v in caption_info.items():
    caption_info_serialize.append({
    'id': i, 
    'caption_pair': k,
    'amr': [a[0] for a in v['amr']], 
    'constituency': ["{}".format(c_trees[0]) for c_trees in v['constituency']],
    'dependency': v['dependency']})
    i+=1

#%%
import json
with open('tree_dump.json','w') as f:
    json.dump(caption_info_serialize,f)
# %%
stog = amrlib.load_stog_model()
graphs = stog.parse_sents(['the person in pink was close to not winning', 'the person in pink was not close to winning'])
for graph in graphs:
    print(graph)

# %%
# visualize the first 35 AMR trees
# defaultdict is ordered from Python3.7
for treeinfo in caption_info_serialize.keys()[:35]:
    amr_1, amr_2 = caption_info[treeinfo]['amr'] 
    plot = AMRPlot()
    plot.build_from_graph(amr_1, debug=False)
    plot.view()   


# %%

# %%
