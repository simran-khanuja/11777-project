from zeno_client import ZenoClient, ZenoMetric
import pandas as pd

# Change this to your API key.
ZENO_API_KEY = "zen_PP8HjtbljTipOMOGBOG2nl6R5IHsSVbxhgOz14vyquw"

client = ZenoClient(ZENO_API_KEY)

# read in your data 
df = pd.read_csv("zeno_code/metadata_group.csv")

# Explicitly save the index as a column to upload.
df["id"] = df.index

df["Avg. AMR Length"] = df["Avg. AMR Length"].astype(int)
df["Avg. Perplexity"] = df["Avg. Perplexity"].astype(float).round(1)

project = client.create_project(
    name="11777 Winoground Group Analysis",
    view="image-classification",
    metrics=[
        ZenoMetric(name="text-retrieval", type="mean", columns=["Group Text Retrieval"]),
        ZenoMetric(name="image-retrieval", type="mean", columns=["Group Image Retrieval"]),
        ZenoMetric(name="image-and-text-retrieval", type="mean", columns=["Group Image-and-Text Retrieval"])
    ]
)

project.upload_dataset(df, id_column="index", data_column='image',
                       label_column="caption")

########################### Unimodal ########################
df_system = pd.read_csv("zeno_code/unimodal_scores/uni-cls.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

# initialize empty columns in df_system
group_text_retrieval = []
group_image_retrieval = []
for i in range(len(df_system["Text Retrieval"])):
    if i%2==0:
        group_text_retrieval.append(df_system["Text Retrieval"][i] & df_system["Text Retrieval"][i+1])
        group_image_retrieval.append(df_system["Image Retrieval"][i] & df_system["Image Retrieval"][i+1])
    else:
        group_text_retrieval.append(df_system["Text Retrieval"][i] & df_system["Text Retrieval"][i-1])
        group_image_retrieval.append(df_system["Image Retrieval"][i] & df_system["Image Retrieval"][i-1])
        
df_system["Group Text Retrieval"] = group_text_retrieval
df_system["Group Image Retrieval"] = group_image_retrieval
df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
df_system = df_system.drop(columns=["Text Retrieval", "Image Retrieval", "Diff from False - Text Retrieval", "Diff from False - Image Retrieval", "True Similarity"])
project.upload_system(df_system, name="RoBERTa-Dinov2 (large)", id_column="id", output_column="Group Image-and-Text Retrieval")

########################### CLIP ########################
df_system = pd.read_csv("zeno_code/clip_scores/clip-g-14.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

# initialize empty columns in df_system
group_text_retrieval = []
group_image_retrieval = []
for i in range(len(df_system["Text Retrieval"])):
    if i%2==0:
        group_text_retrieval.append(df_system["Text Retrieval"][i] & df_system["Text Retrieval"][i+1])
        group_image_retrieval.append(df_system["Image Retrieval"][i] & df_system["Image Retrieval"][i+1])
    else:
        group_text_retrieval.append(df_system["Text Retrieval"][i] & df_system["Text Retrieval"][i-1])
        group_image_retrieval.append(df_system["Image Retrieval"][i] & df_system["Image Retrieval"][i-1])
        
df_system["Group Text Retrieval"] = group_text_retrieval
df_system["Group Image Retrieval"] = group_image_retrieval
df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
df_system = df_system.drop(columns=["Text Retrieval", "Image Retrieval", "Diff from False - Text Retrieval", "Diff from False - Image Retrieval", "True Similarity"])
project.upload_system(df_system, name="CLIP-G-14", id_column="id", output_column="Group Image-and-Text Retrieval")

########################### BLIP2 COCO ########################
df_system = pd.read_csv("zeno_code/blip2scores/blip2coco.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

# initialize empty columns in df_system
group_text_retrieval = []
group_image_retrieval = []
for i in range(len(df_system["Text Retrieval"])):
    if i%2==0:
        group_text_retrieval.append(df_system["Text Retrieval"][i] & df_system["Text Retrieval"][i+1])
        group_image_retrieval.append(df_system["Image Retrieval"][i] & df_system["Image Retrieval"][i+1])
    else:
        group_text_retrieval.append(df_system["Text Retrieval"][i] & df_system["Text Retrieval"][i-1])
        group_image_retrieval.append(df_system["Image Retrieval"][i] & df_system["Image Retrieval"][i-1])
        
df_system["Group Text Retrieval"] = group_text_retrieval
df_system["Group Image Retrieval"] = group_image_retrieval
df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
df_system = df_system.drop(columns=["Text Retrieval", "Image Retrieval", "Diff from False - Text Retrieval", "Diff from False - Image Retrieval", "True Similarity"])
project.upload_system(df_system, name="BLIP-2 COCO", id_column="id", output_column="Group Image-and-Text Retrieval")


########################### BLIP2 Pretrain ########################
# read in your data
df_system = pd.read_csv("zeno_code/blip2scores/blip2pretrain.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

# initialize empty columns in df_system
group_text_retrieval = []
group_image_retrieval = []
for i in range(len(df_system["Text Retrieval"])):
    if i%2==0:
        group_text_retrieval.append(df_system["Text Retrieval"][i] & df_system["Text Retrieval"][i+1])
        group_image_retrieval.append(df_system["Image Retrieval"][i] & df_system["Image Retrieval"][i+1])
    else:
        group_text_retrieval.append(df_system["Text Retrieval"][i] & df_system["Text Retrieval"][i-1])
        group_image_retrieval.append(df_system["Image Retrieval"][i] & df_system["Image Retrieval"][i-1])

df_system["Group Text Retrieval"] = group_text_retrieval
df_system["Group Image Retrieval"] = group_image_retrieval
df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
df_system = df_system.drop(columns=["Text Retrieval", "Image Retrieval", "Diff from False - Text Retrieval", "Diff from False - Image Retrieval", "True Similarity"])

project.upload_system(df_system, name="BLIP-2 Pretrain", id_column="id", output_column="Group Image-and-Text Retrieval")

########################### CLIP Base ########################
# read in your data
df_system = pd.read_csv("wino_clip_base.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

# initialize empty columns in df_system
group_text_retrieval = []
group_image_retrieval = []
for i in range(len(df_system["Text Retrieval"])):
    if i%2==0:
        group_text_retrieval.append(df_system["Text Retrieval"][i] & df_system["Text Retrieval"][i+1])
        group_image_retrieval.append(df_system["Image Retrieval"][i] & df_system["Image Retrieval"][i+1])
    else:
        group_text_retrieval.append(df_system["Text Retrieval"][i] & df_system["Text Retrieval"][i-1])
        group_image_retrieval.append(df_system["Image Retrieval"][i] & df_system["Image Retrieval"][i-1])

df_system["Group Text Retrieval"] = group_text_retrieval
df_system["Group Image Retrieval"] = group_image_retrieval
df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
df_system = df_system.drop(columns=["Text Retrieval", "Image Retrieval", "Diff from False - Text Retrieval", "Diff from False - Image Retrieval", "True Similarity"])

project.upload_system(df_system, name="CLIP-b32", id_column="id", output_column="Group Image-and-Text Retrieval")

########################### ViLT COCO ########################
# read in your data
df_system = pd.read_csv("viltfinetunecoco.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index

# initialize empty columns in df_system
group_text_retrieval = []
group_image_retrieval = []
for i in range(len(df_system["Text Retrieval"])):
    if i%2==0:
        group_text_retrieval.append(df_system["Text Retrieval"][i] & df_system["Text Retrieval"][i+1])
        group_image_retrieval.append(df_system["Image Retrieval"][i] & df_system["Image Retrieval"][i+1])
    else:
        group_text_retrieval.append(df_system["Text Retrieval"][i] & df_system["Text Retrieval"][i-1])
        group_image_retrieval.append(df_system["Image Retrieval"][i] & df_system["Image Retrieval"][i-1])

df_system["Group Text Retrieval"] = group_text_retrieval
df_system["Group Image Retrieval"] = group_image_retrieval
df_system["Group Image-and-Text Retrieval"] = df_system["Group Image Retrieval"] & df_system["Group Text Retrieval"]

# remove text and image retrieval columns
df_system = df_system.drop(columns=["Text Retrieval", "Image Retrieval", "Diff from False - Text Retrieval", "Diff from False - Image Retrieval", "True Similarity"])

project.upload_system(df_system, name="ViLT-b32 COCO", id_column="id", output_column="Group Image-and-Text Retrieval")