import json
from PIL import Image

from tqdm import tqdm
import pandas as pd

import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from transformers import CLIPProcessor, CLIPModel

from custom_batching import group_data_by_id, GroupedBatchSampler

# Define a custom dataset
class image_text_dataset():
    def __init__(self, list_image_path, list_txt, group_ids):
        # Initialize image paths and corresponding texts
        self.group_ids = group_ids
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.text = clip.tokenize([text.lower().replace(" .","") for text in list_txt])

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # print(idx)
        if isinstance(idx, tuple):
            # Process batch of indices
            images = []
            texts = []
            group_ids = []
            for i in idx:
                image = preprocess(Image.open(self.image_path[i]))
                text = self.text[i]
                # clip_processor(text=[winoground[155]["caption_1"]], images=[winoground[155]["image_0"].convert("RGB")], return_tensors="pt")
                # image = processor(images=[Image.open(self.image_path[i]).convert("RGB")], return_tensors="pt").pixel_values
                images.append(image)
                texts.append(text)
                group_ids.append(self.group_ids[i])

            return images, texts, group_ids
        else:
            # Process single index
            image = preprocess(Image.open(self.image_path[idx]))
            text = self.text[idx]
            group_id = self.group_ids[idx]
            return image, text, group_id

if __name__ == "__main__":
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Choose computation device
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    # Load pre-trained CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    print(device)
    # use your own data
    list_image_path = []
    list_txt = []
    group_ids = []

    metadata = pd.read_csv("metadata.csv")

    original_image_paths = metadata["image_path"].tolist()
    original_captions = metadata["caption"].tolist()
    augmented_image_paths = metadata["augmented_image_path"].tolist()
    augmented_captions = metadata["augmented_caption"].tolist()

    for i in range(len(original_image_paths)):
        list_image_path.append(original_image_paths[i])
        list_txt.append(original_captions[i])
        group_ids.append(i)

        list_image_path.append(augmented_image_paths[i])
        list_txt.append(augmented_captions[i])
        group_ids.append(i)

    dataset = image_text_dataset(list_image_path, list_txt, group_ids)
    # Assuming 'dataset' is your original dataset
    grouped_data = group_data_by_id(dataset)
    # print(grouped_data.keys())

    grouped_batch_sampler = GroupedBatchSampler(grouped_data, batch_size=512)
    train_dataloader = DataLoader(dataset, batch_sampler=grouped_batch_sampler)

    # Function to convert model's parameters to FP32 format
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 

    if device == "cpu":
        model.float()

    # Prepare the optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.1) # the lr is smaller, more safe for fine tuning to new dataset

    # Specify the loss function
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch in pbar:
            optimizer.zero_grad()

            images,texts,_ = batch 
            
            images= images.to(device)
            texts = texts.to(device)

            # print(images.shape)
            # print(texts.shape)

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # Compute loss
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss = 0.7 * loss_img(logits_per_image,ground_truth) + 0.3 * loss_txt(logits_per_text,ground_truth)

            # Backward pass
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else: 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
    
    # Save the model
    torch.save(model.state_dict(), "clip-ft/model_5_2e5_512-07.pt")
