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
from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.
#%%

#%%

cuda_device = args.device
def dist_ex(rank, world_size):
    print(rank, world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    with open('augmented_images_train.pkl', 'rb') as f:
        dataset = pickle.load(f)
        dataset = dataset.filter(lambda x: x["label"] == 1)

    with open('augmented_images_val.pkl', 'rb') as f:
        dataset_val = pickle.load(f)
        dataset_val = dataset_val.filter(lambda x: x["label"] == 1)
    #%%
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    model, vis_processors, text_processors = load_model_and_preprocess("blip2", "coco", device=device)

    #%%
    print(model.device)
    #%%
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F

    class ImageToTextMatchingDataset(Dataset):
        def __init__(self, dataset, vis_processor, text_processor, device):
            self.dataset = dataset
            self.vis_processor = vis_processor
            self.text_processor = text_processor
            self.device = device

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            # remove batch dimension
            encoding = {"image": self.vis_processor(item["image"]).to(self.device)}
            encoding["text_input"] = self.text_processor(item["caption"])
            # encoding["label"] = torch.tensor(item["label"], dtype=torch.float).unsqueeze(0).to(self.device)
            return encoding



    def collate_fn(batch):
        # pad the input_ids and attention_mask
        processed_batch = {}
        # print(batch)
        for key in batch[0].keys():
            if key != "text_input":
                processed_batch[key] = torch.stack([example[key] for example in batch]).to(device)
            else:
                processed_batch[key] = [example[key] for example in batch]
        return processed_batch

    def eval_collate_fn(batch):
        # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text_input":
                processed_batch[key] = torch.stack([example[key] for example in batch]).to(device)
            else:
                processed_batch[key] = [example[key] for example in batch]
        return processed_batch

    #%%
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    device = torch.device(cuda_device) if torch.cuda.is_available() else "cpu"
    train_dataset = ImageToTextMatchingDataset(dataset.select(range(len(dataset)//16*16)), vis_processors["train"], text_processors['train'], device)
    val_dataset = ImageToTextMatchingDataset(dataset_val, vis_processors["eval"], text_processors['eval'], device)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, collate_fn=eval_collate_fn)
    #%%
    # from peft import LoraConfig, get_peft_model

    # # Let's define the LoraConfig
    # config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     target_modules=["q_proj", "k_proj"]
    # )

    # model = get_peft_model(model, config)
    # print(list(model.parameters()))
    #%%


    #%%
    # from torchviz import make_dot

    # batch = next(iter(train_dataloader))
    # yhat = model(batch, match_head="itc")
    # print(yhat, batch["label"])
    # loss = loss_fn(yhat, batch["label"])
    # make_dot(loss, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    #%%
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
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


            if (idx+1) % args.eval_steps == 0:
                ddp_model.eval()
                with torch.no_grad():
                    val_loss = []
                    for v_idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                        outputs = ddp_model(batch)
                        # print(outputs)
                        loss = outputs.loss_itc
                        val_loss.append(loss) 
                ddp_model.train()
                print("Val Loss:", sum(val_loss) / len(val_loss))
                wandb.log({f"val_loss_{rank}": sum(val_loss) / len(val_loss)}, step= epoch * len(train_dataloader) + idx)
        if rank == 0:
            torch.save(ddp_model.module.state_dict(), f"blip2_epoch_{epoch}.pth")
    ddp_model.eval()
    with torch.no_grad():
        val_loss = []
        for v_idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            outputs = ddp_model(batch)
            # print(outputs)
            loss = outputs.loss_itc
            val_loss.append(loss) 
    print("Val Loss:", sum(val_loss) / len(val_loss))

    wandb.finish()
    torch.save(ddp_model.module.state_dict(), "blip2.pth")
# torch.save(model.state_dict(), "blip2_image_text_matching_pretrain.pth")
#%%

#%%
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

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