#%%
import torch
from PIL import Image
import wandb
from lavis.models import load_model_and_preprocess, load_preprocess
from lavis.processors import load_processor
from lavis.models.blip2_models.blip2_image_text_matching import Blip2ITM
import matplotlib.pyplot as plt
import pickle
from transformers import AutoProcessor


import argparse
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from torch.utils.data import Sampler
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os
#%%
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--eval_steps", type=int, default=10)
args = parser.parse_args()
print(args)

wandb.init(project="blip2contrastiveimageaug", entity="aethersura", config= { "batch_size": args.batch_size, "num_epochs": args.num_epochs, "lr": args.lr})


#%%


np.random.seed(42)
def group_data_by_id(dataset):
    grouped_data = defaultdict(list)
    for i in range(len(dataset)):
        _, _, group_id = dataset[i]
        grouped_data[group_id].append(i)
    return grouped_data

class GroupedBatchSampler(Sampler):
    def __init__(self, grouped_data, batch_size):
        self.grouped_data = grouped_data
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def _create_batches(self):
        batches = []
        group_indices = list(self.grouped_data.keys())
        # Shuffle group indices
        np.random.shuffle(group_indices)
        # print("group_indices: ", group_indices)

        for i in range(0, len(group_indices), self.batch_size // 2):
            batch = []
            for group_idx in group_indices[i:i+self.batch_size // 2]:
                for data in self.grouped_data[group_idx]:
                    batch.append(data)
            batches.append(batch)
        
        # print(batch)
        # Shuffle batches
        np.random.shuffle(batches)
        return batches

    def __iter__(self):
        for batch in self.batches:

            yield batch

    def __len__(self):
        return len(self.batches)

# Define a custom dataset
class image_text_dataset():
    def __init__(self, list_image_path, list_txt, group_ids, vis_processor, text_processor, device):
        # Initialize image paths and corresponding texts
        self.group_ids = group_ids
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.text = [text.lower().replace(" .","") for text in list_txt]
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.device = device

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
                image = self.vis_processor(Image.open(self.image_path[i])).to(self.device)
                text = self.text_processor(self.text[i])
                # clip_processor(text=[winoground[155]["caption_1"]], images=[winoground[155]["image_0"].convert("RGB")], return_tensors="pt")
                # image = processor(images=[Image.open(self.image_path[i]).convert("RGB")], return_tensors="pt").pixel_values
                images.append(image)
                texts.append(text)
                group_ids.append(self.group_ids[i])

            return images, texts, group_ids
        else:
            # Process single index
            image = self.vis_processor(Image.open(self.image_path[idx])).to(self.device)
            text = self.text_processor(self.text[idx])
            group_id = self.group_ids[idx]
            return image, text, group_id

device = args.device
def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {"image": torch.stack([image for image, text, group_id in batch]).to(device),
                              "text_input": [text for image, text, group_id in batch]}

    return processed_batch



#%%



cuda_device = args.device
def dist_ex(rank, world_size):
    print(rank, world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    metadata = pd.read_csv("metadata.csv")

    original_image_paths = metadata["image_path"].tolist()
    original_captions = metadata["caption"].tolist()
    augmented_image_paths = metadata["augmented_image_path"].tolist()
    augmented_captions = metadata["augmented_caption"].tolist()

    list_image_path = []
    list_txt = []
    group_ids = []
    for i in range(len(original_image_paths)):
        list_image_path.append(original_image_paths[i])
        list_txt.append(original_captions[i])
        group_ids.append(i)

        list_image_path.append(augmented_image_paths[i])
        list_txt.append(augmented_captions[i])
        group_ids.append(i)


    # print(grouped_data.keys())
    #%%
    device = torch.device(args.device) if torch.cuda.is_available() else "cpu"

    model, vis_processors, text_processors = load_model_and_preprocess("blip2", "coco", device=device)

    dataset = image_text_dataset(list_image_path, list_txt, group_ids, vis_processors["train"], text_processors["train"], device)
    # Assuming 'dataset' is your original dataset
    grouped_data = group_data_by_id(dataset)
    grouped_batch_sampler = GroupedBatchSampler(grouped_data, batch_size=args.batch_size)
    train_dataloader = DataLoader(dataset, batch_sampler=grouped_batch_sampler, collate_fn=collate_fn)

    #%%
    print(model.device)
    #%%
    #%%
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    device = torch.device(args.device) if torch.cuda.is_available() else "cpu"
    
    ddp_model = DDP(model, device_ids=[rank])
    # Prepare the optimizer
    optimizer = torch.optim.Adam(ddp_model.parameters(),lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.1) # the lr is smaller, more safe for fine tuning to new dataset

    
    #%%
    ddp_model.train()
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # batch['image'] = batch['image'].to(device)
            outputs = ddp_model(batch)
            loss = outputs.loss
            wandb.log({f"train_loss_{rank}": loss},step= epoch * len(train_dataloader) + idx)
            # print("Loss:", loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if (idx+1) % args.eval_steps == 0:
            #     ddp_model.eval()
            #     with torch.no_grad():
            #         val_loss = []
            #         for v_idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            #             outputs = ddp_model(batch)
            #             # print(outputs)
            #             loss = outputs.loss_itc
            #             val_loss.append(loss) 
            #     ddp_model.train()
            #     print("Val Loss:", sum(val_loss) / len(val_loss))
            #     wandb.log({f"val_loss_{rank}": sum(val_loss) / len(val_loss)}, step= epoch * len(train_dataloader) + idx)
        if rank == 0:
            torch.save(ddp_model.module.state_dict(), f"blip2_epoch_{epoch}.pth")


    wandb.finish()
    torch.save(ddp_model.module.state_dict(), "blip2.pth")
# torch.save(model.state_dict(), "blip2_image_text_matching_pretrain.pth")
#%%

#%%

def main():
    world_size = 1
    mp.spawn(dist_ex,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()