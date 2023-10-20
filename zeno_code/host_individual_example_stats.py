from zeno_client import ZenoClient, ZenoMetric
import pandas as pd

# Change this to your API key.
ZENO_API_KEY = "zen_PP8HjtbljTipOMOGBOG2nl6R5IHsSVbxhgOz14vyquw"

client = ZenoClient(ZENO_API_KEY)

# read in your data 
df = pd.read_csv("metadata_individual_example.csv")

# Explicitly save the index as a column to upload.
df["id"] = df.index

# Convert the columns to the correct types.
df["AMR Length"] = df["AMR Length"].astype(int)
df["No. of Obj-Relations"] = df["No. of Obj-Relations"].astype(int)
df["GPT2 Perplexity"] = df["GPT2 Perplexity"].astype(float).round(1)

# Create Project
project = client.create_project(
    name="Winoground Model Analysis",
    view="image-classification",
    metrics=[
        ZenoMetric(name="text-retrieval", type="mean", columns=["Text Retrieval"]),
        ZenoMetric(name="image-retrieval", type="mean", columns=["Image Retrieval"]),
        ZenoMetric(name="image-and-text-retrieval", type="mean", columns=["Image-and-Text Retrieval"])
    ]
)

project.upload_dataset(df, id_column="index", data_column='image',
                       label_column="caption")

######################## BLIP2 VQA Scores (is honest) ########################
df_system = pd.read_csv("simran/blip2_vqascore_is_honest.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index
df_system["Image-and-Text Retrieval"] = df_system["Image Retrieval"] & df_system["Text Retrieval"]
project.upload_system(df_system, name="BLIP-2 VQA (is honest)", id_column="id", output_column="Image-and-Text Retrieval")


######################## BLIP2 COCO ITM ########################
# read in your data
df_system = pd.read_csv("simran/blip2_coco.csv")


# Create an id column to match the base dataset.
df_system["id"] = df_system.index
df_system["Image-and-Text Retrieval"] = df_system["Image Retrieval"] & df_system["Text Retrieval"]
project.upload_system(df_system, name="BLIP-2 COCO ITM", id_column="id", output_column="Image-and-Text Retrieval")

######################## BLIP2 VQA ########################
# read in your data
df_system = pd.read_csv("simran/blip2_vqa.csv")

# Create an id column to match the base dataset.
df_system["id"] = df_system.index
df_system["Image-and-Text Retrieval"] = df_system["Image Retrieval"] & df_system["Text Retrieval"]
project.upload_system(df_system, name="BLIP-2 VQA", id_column="id", output_column="Image-and-Text Retrieval")