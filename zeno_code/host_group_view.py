from zeno_client import ZenoClient, ZenoMetric
import pandas as pd
import json

# Change this to your API key.
ZENO_API_KEY = "zen_PP8HjtbljTipOMOGBOG2nl6R5IHsSVbxhgOz14vyquw"

client = ZenoClient(ZENO_API_KEY)

# read in your data 
df = pd.read_csv("zeno_code/metadata_group_view.csv")

# Explicitly save the index as a column to upload.
df["id"] = df.index

df["Avg. AMR Length"] = df["Avg. AMR Length"].astype(int)
df["Avg. Perplexity"] = df["Avg. Perplexity"].astype(float).round(1)
# Format each element in the list with double quotes
df['image'] = df[['image1', 'image2']].apply(lambda x: f'["{x[0]}", "{x[1]}"]', axis=1)
df['caption'] = df[['caption1', 'caption2']].apply(lambda x: f'["{x[0]}", "{x[1]}"]', axis=1)

# Remove the original columns
df = df.drop(columns=['image1', 'image2', 'caption1', 'caption2'])

# Example of how it's stored and printed
print(df['image'][0])
print(df['caption'][0])


project = client.create_project(
    name="11777 Winoground Group View Analysis",
    view={
        "data": {
            "type": "list",
            "elements": {
            "type": "image"
            }
        },
        "label": {
            "type": "list",
            "elements": {
            "type": "text"
            }
        },
        "output": {
            "type": "text"
        }
    },
    metrics=[
        ZenoMetric(name="text-retrieval", type="mean", columns=["Group Text Retrieval"]),
        ZenoMetric(name="image-retrieval", type="mean", columns=["Group Image Retrieval"]),
        ZenoMetric(name="image-and-text-retrieval", type="mean", columns=["Group Image-and-Text Retrieval"])
    ]
)

project.upload_dataset(df, id_column="index", data_column="image",
                       label_column="caption")

########################### Unimodal ########################
df_system = pd.read_csv("zeno_code/unimodal_scores/uni-cls_grp.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index
df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
project.upload_system(df_system, name="RoBERTa-Dinov2 (large)", id_column="id", output_column="Group Image-and-Text Retrieval")

########################### CLIP ########################
df_system = pd.read_csv("zeno_code/clip_scores/clip-g-14_grp.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index
df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
project.upload_system(df_system, name="CLIP-G-14", id_column="id", output_column="Group Image-and-Text Retrieval")

########################### BLIP2 COCO ########################
df_system = pd.read_csv("zeno_code/blip2scores/blip2coco_grp.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
project.upload_system(df_system, name="BLIP-2 COCO", id_column="id", output_column="Group Image-and-Text Retrieval")


########################### BLIP2 Pretrain ########################
# read in your data
df_system = pd.read_csv("zeno_code/blip2scores/blip2pretrain_grp.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
project.upload_system(df_system, name="BLIP-2 Pretrain", id_column="id", output_column="Group Image-and-Text Retrieval")

########################### CLIP Base ########################
# read in your data
df_system = pd.read_csv("wino_clip_base_grp.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
project.upload_system(df_system, name="CLIP-b32", id_column="id", output_column="Group Image-and-Text Retrieval")

########################### ViLT COCO ########################
# read in your data
df_system = pd.read_csv("viltfinetunecoco_grp.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
project.upload_system(df_system, name="ViLT-b32 COCO", id_column="id", output_column="Group Image-and-Text Retrieval")

########################### BLIP base ########################
# read in your data
df_system = pd.read_csv("zeno_code/blip_scores/blipbase_grp.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
project.upload_system(df_system, name="BLIP base", id_column="id", output_column="Group Image-and-Text Retrieval")


########################### BLIP large ########################
# read in your data
df_system = pd.read_csv("zeno_code/blip_scores/bliplarge_grp.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
project.upload_system(df_system, name="BLIP large", id_column="id", output_column="Group Image-and-Text Retrieval")

########################### CLIP Synthetic ########################
# read in your data
df_system = pd.read_csv("zeno_code/wino_synthetic_group.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
project.upload_system(df_system, name="CLIP base synthetic", id_column="id", output_column="Group Image-and-Text Retrieval")

########################### BLIP-2 Synthetic ########################
# read in your data
df_system = pd.read_csv("zeno_code/blip2_coco_syn_grp.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
project.upload_system(df_system, name="BLIP2 COCO synthetic", id_column="id", output_column="Group Image-and-Text Retrieval")


########################### LLaVa ########################
# read in your data
df_system = pd.read_csv("zeno_code/llava_scores/llava_grp.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
project.upload_system(df_system, name="LLava-1.5-13b", id_column="id", output_column="Group Image-and-Text Retrieval")
