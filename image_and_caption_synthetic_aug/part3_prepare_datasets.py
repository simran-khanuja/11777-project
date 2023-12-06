#%%
from datasets import Dataset, concatenate_datasets
import pickle
from PIL import Image
from tqdm import tqdm

#%%
# with open("filtered_dataset_step_3_part1.pkl", "rb") as f:
#     dataset_1 = pickle.load(f)

# with open("filtered_dataset_step_3_part2.pkl", "rb") as f:
#     dataset_2 = pickle.load(f)

with open("filtered_dataset_step_3.pkl", "rb") as f:
    dataset = pickle.load(f)

# dataset = concatenate_datasets([dataset_1, dataset_2, dataset])
#%%
# save the image and aug images in a folder
# save the captions in a csv file
image_paths = []
aug_image_paths = []
captions = []
aug_captions = []
for data in tqdm(dataset):
    image = data['image']
    image_path = "images_folder/" + str(data['imgid']) + ".jpg"
    image.save(image_path)
    aug_image = data['aug_image']
    aug_image_path = "images_folder/" + str(data['imgid']) + "_aug.jpg"
    aug_image.save(aug_image_path)
    image_paths.append(image_path)
    aug_image_paths.append(aug_image_path)
    captions.append(data['caption_pair'][0].strip('.').strip().lower())
    aug_captions.append(data['caption_pair'][1].strip('.').strip().lower())

import pandas as pd
df = pd.DataFrame({"image_path": image_paths, "aug_image_path": aug_image_paths, "caption": captions, "aug_caption": aug_captions})
df.to_csv("captions.csv", index=False)
