# %%
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from datasets import load_dataset
import stanza
from collections import defaultdict
from tqdm import tqdm
import amrlib
import spacy
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
    doc1 = conParser(tup[0])
    doc2 = conParser(tup[1])
    doc1_s = nlp(tup[0])
    doc2_s = nlp(tup[1])
    doc1_cons = [sentence.constituency for sentence in doc1.sentences]
    doc2_cons = [sentence.constituency for sentence in doc2.sentences]

    doc1_deps = [{'id':word.id, 'word': word.text, 'head id': word.head, 'head': sent.words[word.head-1].text if word.head > 0 else "root", 'deprel': word.deprel} for sent in doc1.sentences for word in sent.words]
    doc2_deps = [{'id':word.id, 'word': word.text, 'head id': word.head, 'head': sent.words[word.head-1].text if word.head > 0 else "root", 'deprel': word.deprel} for sent in doc2.sentences for word in sent.words]

    doc1_graphs = stog.parse_sents([sent.text for sent in doc1_s.sents])
    doc2_graphs = stog.parse_sents([sent.text for sent in doc2_s.sents])

    caption_info[(tup[0],tup[1])]['constituency'] = [doc1_cons, doc2_cons]
    caption_info[(tup[0],tup[1])]['dependency'] = [doc1_deps, doc2_deps]
    caption_info[(tup[0],tup[1])]['amr'] = [doc1_graphs, doc2_graphs]

# %%
caption_info[('the person in pink was close to not winning', 'the person in pink was not close to winning')]
# %%
# write 200 lines into file: 100 caption0s and 100 caption1s
with open('camr_word_list.txt','w',encoding='utf-8-sig') as f:
    f.writelines([caption.strip()+'\n' for caption in caption_0s])
    f.writelines([caption.strip()+'\n' for caption in caption_1s])
# %%
stog = amrlib.load_stog_model()
graphs = stog.parse_sents(['the person in pink was close to not winning', 'the person in pink was not close to winning'])
for graph in graphs:
    print(graph)
# %%
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# %%
