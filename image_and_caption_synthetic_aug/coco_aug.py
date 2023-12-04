# %%
import nltk
import spacy
import math
import webcolors
import pickle
import torch
import numpy as np
import torch.nn.functional as F
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets
from collections import defaultdict
from itertools import permutations
from sentence_transformers import SentenceTransformer
# %%
nlp = spacy.load('en_core_web_lg')
# %%
# iterate over dataset and find captions per image
def filter_dataset_2_adj_color(dataset):
    filter_indices = []
    length = len(dataset)
    for idx, data in tqdm(enumerate(dataset), total=length):
        caption = data['sentences']['raw']
        doc = nlp(caption)
        pos_tags = [token.pos_ for token in doc]
        # check if they're present in the pos_tags_list
        # if pos_tags in pos_tags_list:
        #     filtered_dataset.append(data)
        #     continue
        # else check if exactly 2 noun chunks with color adjectives in both of them
        noun_chunks = list(doc.noun_chunks)
        if len(noun_chunks) == 2:
            # check if both noun chunks have 1 adjective
            adj_flag = True
            for chunk in noun_chunks:
                c_adj = 0
                for token in chunk:
                    if token.pos_ == 'ADJ' and token.text.lower() in webcolors.CSS3_NAMES_TO_HEX:
                        c_adj += 1 
                if c_adj != 1:
                    adj_flag = False
                    break
            if adj_flag:
                filter_indices.append(data['sentences']['sentid'])
                continue
    # filter dataset 
    filtered_dataset = dataset.filter(lambda x: x['sentences']['sentid'] in filter_indices)
    return filtered_dataset

# %%
def filter_data_step0_multiprocess(dataset):
    # create multiprocessed function to iterate over dataset and find captions per image
    num_processes = multiprocessing.cpu_count()//2
    pool = Pool(num_processes)

    # split dataset into chunks
    data_mscoco = dataset['train']
    len_data = len(data_mscoco)
    chunk_size = math.ceil(len_data/(num_processes))

    chunks = [data_mscoco.select(range(i,min(i + chunk_size,len_data))) for i in range(0, len_data, chunk_size)]
    filtered_dataset = pool.map(filter_dataset_2_adj_color, chunks)
    # flatten the list

    filtered_dataset = concatenate_datasets(filtered_dataset)
    return filtered_dataset
# %%
# get unique values by 'imgid' 
def filter_data_step1_dump(filtered_dataset):
    unique_imgid = defaultdict(list)
    for data in tqdm(filtered_dataset):
        unique_imgid[data['imgid']] = data

    # combine all these rows into a dataset

    unique_imgid = list(unique_imgid.values())

    filtered_dataset = Dataset.from_list(unique_imgid)

    print(f"There are {len(filtered_dataset)} datapoints of MSCOCO captions after filtering")
    
    with open('1filtered_dataset.pkl', 'wb') as f:
        pickle.dump(filtered_dataset, f)
    return filtered_dataset

# %%
# import pickle
# with open('filtered_dataset.pkl', 'rb') as f:
#     filtered_dataset = pickle.load(f)

# %%
def swap_adjectives_across_noun_chunks(doc: spacy.tokens.doc.Doc):
    adj_pos = []
    adj_word = []
    token_list = [token for token in doc]
    for chunk in doc.noun_chunks:
        # get the adjective 
        for token in chunk:
            if token.pos_ == 'ADJ' and token.text.lower() in webcolors.CSS3_NAMES_TO_HEX:
                adj_pos.append(token.i)
                adj_word.append(token)
                break
    perms = list(permutations(adj_word))
    sentences = []
    for perm in perms:
        # fill out all the tokens
        new_token_list = token_list
        for i, pos in enumerate(adj_pos):
            new_token_list[pos] = perm[i]
        sentences.append(new_token_list.copy())
    return sentences

# %%
def filter_by_similarity(filtered_dataset):
    minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    filtered_caption_similarities = []
    similarities = []
    for data in tqdm(filtered_dataset):
        # print(data)
        captions = swap_adjectives_across_noun_chunks(nlp(data['sentences']['raw']))
        text_captions = [" ".join(token.text for token in caption) for caption in captions]
        # print(text_captions)
        caption_pair_embeds = minilm.encode(text_captions)
        similarity = F.cosine_similarity(torch.tensor(caption_pair_embeds[0]), torch.tensor(caption_pair_embeds[1]), dim=0)
        similarities.append(similarity)
        if similarity > 0.97:
            # append all captions to datapoint
            if text_captions[0].lower() == text_captions[1].lower():
                continue
            data['caption_pair'] = text_captions
            filtered_caption_similarities.append(data)
    filtered_dataset_step_2 = Dataset.from_list(filtered_caption_similarities)
    print(f"There are {len(filtered_dataset_step_2)} datapoints of MSCOCO captions after filtering")
    with open('1filtered_dataset_step_2.pkl', 'wb') as f:
        pickle.dump(filtered_dataset_step_2, f)
    return filtered_dataset_step_2

#%%
dataset = load_dataset('HuggingFaceM4/COCO')
dataset['train'] = dataset['train'].sort('imgid')

filtered_dataset = filter_data_step0_multiprocess(dataset)
filtered_dataset = filter_data_step1_dump(filtered_dataset)
with open('1filtered_dataset.pkl', 'rb') as f:
    filtered_dataset = pickle.load(f)
filtered_dataset_step_2 = filter_by_similarity(filtered_dataset)

# %%
# with open('filtered_dataset_step_2.pkl', 'rb') as f:
#     filtered_dataset_step_2 = pickle.load(f)

# # %%
# filtered_dataset_step_2.select([0])[0]['image']

# image
# white dog and white and blue umbrella
# blue dog and white and white umbrella
# image_2