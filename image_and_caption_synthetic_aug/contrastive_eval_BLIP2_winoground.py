#%%
import torch
from PIL import Image
import wandb
from lavis.models import load_model_and_preprocess, load_preprocess
from lavis.processors import load_processor
from lavis.models.blip2_models.blip2_image_text_matching import Blip2ITM
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import AutoProcessor
import torch
from tqdm import tqdm
import argparse
import torch.distributed as dist
import gc
import json
#%%
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--device", type=str, default="cuda:2")
parser.add_argument("--ckpt", type=str, default="/usr1/data/abhinavr/blip2_batch_epoch_1.pth")
args = parser.parse_args()
print(args)
print(torch.cuda.device_count())
wandb.init(project="blip2contrastive_eval", entity="aethersura", config= { "batch_size": args.batch_size, "num_epochs": args.num_epochs, "lr": args.lr})

#%%
from torch.nn.parallel import DistributedDataParallel as DDP
#%%

# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "29500"

device = torch.device(args.device) if torch.cuda.is_available() else "cpu"
# dist.init_process_group("gloo", rank=1, world_size=1)
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)
model.load_state_dict(torch.load(args.ckpt))

#%%
model
#%%
from datasets import load_dataset
auth_token = "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"  # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]

def transform_wino(examples):
    examples["image_0"] = [image.convert("RGB") for image in examples["image_0"]]
    examples["image_1"] = [image.convert("RGB") for image in examples["image_1"]]
    return examples

winoground.set_transform(transform_wino)

blip2_coco_scores = []

for sample in tqdm(winoground):
    image_0, cap_0 = sample["image_0"], sample["caption_0"]
    image_1, cap_1 = sample["image_1"], sample["caption_1"]

    # Preprocess the images
    image_0 = vis_processors["eval"](image_0).unsqueeze(0).to(device)
    cap_0 = text_processors["eval"](cap_0)
    image_1 = vis_processors["eval"](image_1).unsqueeze(0).to(device)
    cap_1 = text_processors["eval"](cap_1)

    im0_cap0_match = model({"image": image_0, "text_input": cap_0}, match_head="itm")
    
    im0_cap0_scores = torch.nn.functional.softmax(im0_cap0_match, dim=1)
    # print(im0_cap0_match, im0_cap0_scores)
    im0_cap1_match = model({"image": image_0, "text_input": cap_1}, match_head="itm")
    im0_cap1_scores = torch.nn.functional.softmax(im0_cap1_match, dim=1)

    im1_cap0_match = model({"image": image_1, "text_input": cap_0}, match_head="itm")
    im1_cap0_scores = torch.nn.functional.softmax(im1_cap0_match, dim=1)

    im1_cap1_match = model({"image": image_1, "text_input": cap_1}, match_head="itm")
    im1_cap1_scores = torch.nn.functional.softmax(im1_cap1_match, dim=1)

    # Get the probability scores
    im0_cap0_scores = im0_cap0_scores[:, 1].item()
    im0_cap1_scores = im0_cap1_scores[:, 1].item()
    im1_cap0_scores = im1_cap0_scores[:, 1].item()
    im1_cap1_scores = im1_cap1_scores[:, 1].item()

    blip2_coco_scores.append({"id" : sample["id"], "c0_i0": im0_cap0_scores, "c0_i1": im1_cap0_scores, "c1_i0": im0_cap1_scores, "c1_i1": im1_cap1_scores})


def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

def group_correct(result):
    return image_correct(result) and text_correct(result)

text_correct_count = 0
image_correct_count = 0
group_correct_count = 0
for result in blip2_coco_scores:
  text_correct_count += 1 if text_correct(result) else 0
  image_correct_count += 1 if image_correct(result) else 0
  group_correct_count += 1 if group_correct(result) else 0

num_samples = len(winoground)
print("text score:", text_correct_count/num_samples)
print("image score:", image_correct_count/num_samples)
print("group score:", group_correct_count/num_samples)

wino_blip2_coco = {str(d['id']): {k: v for k, v in d.items() if k != 'id'} for d in blip2_coco_scores}
wino_blip2_coco_json = json.dumps(wino_blip2_coco)
with open("wino_blip2_coco_aug.json", "w") as f:
    f.write(wino_blip2_coco_json)

#%%

    #%%
    # ddp_model = DDP(model, device_ids=[rank])
    # optimizer = torch.optim.Adam(ddp_model.parameters(), lr=5e-4)
    #%%

# torch.save(model.state_dict(), "blip2_image_text_matching_pretrain.pth")
#%%

#%%
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def main():
    pass

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()