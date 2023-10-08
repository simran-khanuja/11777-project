# %%
from datasets import load_dataset
import stanza
from collections import defaultdict
from tqdm import tqdm
import amrlib
import spacy
from   amrlib.graph_processing.amr_plot import AMRPlot
import json
import pydotplus
# %%
dataset = load_dataset('facebook/winoground',use_auth_token=True)
# %%
winoground_text = dataset['test'].select_columns(['caption_0','caption_1'])
# %%
conParser = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,constituency,depparse', use_gpu=True)
depParser = spacy.load('en_core_web_sm')
stog = amrlib.load_stog_model(device='cuda:0')

# %%
caption_info = defaultdict(dict)
doc1_sentences = doc2_sentences = []

for tup in tqdm(winoground_text):
    # create stanza and SpaCy objects
    doc1_c = conParser(tup['caption_0'])
    doc2_c = conParser(tup['caption_1'])

    doc1_d = depParser(tup['caption_0'])
    doc2_d = depParser(tup['caption_1'])

    # extract constituency tree
    doc1_cons = [sentence.constituency for sentence in doc1_c.sentences] # use the read_tree method in here: https://github.com/stanfordnlp/stanza/blob/main/stanza/models/constituency/tree_reader.py to get it back
    doc2_cons = [sentence.constituency for sentence in doc2_c.sentences]

    # extract dependency tree
    doc1_deps = [{'id':word.id, 'word': word.text, 'head id': word.head, 'head': sent.words[word.head-1].text if word.head > 0 else "root", 'deprel': word.deprel} for sent in doc1_c.sentences for word in sent.words]
    doc2_deps = [{'id':word.id, 'word': word.text, 'head id': word.head, 'head': sent.words[word.head-1].text if word.head > 0 else "root", 'deprel': word.deprel} for sent in doc2_c.sentences for word in sent.words]

    # extract AMR tree
    doc1_graphs = stog.parse_sents([sent.text for sent in doc1_d.sents])
    doc2_graphs = stog.parse_sents([sent.text for sent in doc2_d.sents])

    # add to caption
    caption_info[(tup['caption_0'],tup['caption_1'])]['constituency'] = [doc1_cons, doc2_cons]
    caption_info[(tup['caption_0'],tup['caption_1'])]['dependency'] = [doc1_deps, doc2_deps]
    caption_info[(tup['caption_0'],tup['caption_1'])]['amr'] = [doc1_graphs, doc2_graphs]

# %%
# serialize it into something readable
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

# %% 
# dump serialized file
with open('tree_dump.json','w') as f:
    json.dump(caption_info_serialize,f)

#%%
caption_info_serialize = ""
with open('tree_dump.json','r') as f:
     caption_info_serialize = f.read()
     caption_info_serialize = json.loads(caption_info_serialize)
# %%
# visualize the first 35 AMR trees
# tried to render them in the same file, but it proved to be quite hard
caption_info_serialize.sort(key= lambda x: x['id'])

for treeinfo in caption_info_serialize[:35]:
    amr_1, amr_2 = treeinfo['amr'] 
    c_1, c_2 = treeinfo['caption_pair']
    
    plot = AMRPlot()
    plot.build_from_graph(amr_1, debug=False)
    plot2 = AMRPlot()
    plot2.build_from_graph(amr_2, debug=False)

    plot.graph.render(f'amr_plots/{treeinfo["id"]}/{"_".join(c_1.split())}',format='png')
    plot2.graph.render(f'amr_plots/{treeinfo["id"]}/{"_".join(c_2.split())}',format='png')
