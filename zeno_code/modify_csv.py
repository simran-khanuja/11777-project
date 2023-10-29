import pandas as pd

# Load the CSV file
df = pd.read_csv("zeno_code/metadata_group.csv")

# Create an empty list to store the new rows
new_rows = []

# Process every two rows together
for i in range(0, len(df), 2):
    row1 = df.iloc[i]
    row2 = df.iloc[i + 1]

    # Combine the relevant data into a dictionary
    new_row = {
        "index": row1["index"] // 2,
        "image1": row1["image"],
        "caption1": row1["caption"],
        "image2": row2["image"],
        "caption2": row2["caption"],
        "Intra-img Similarity": row1["Intra-img Similarity"],
        "Intra-caption Similarity": row1["Intra-caption Similarity"],
        "Avg. Phrase Grounding": row1["Avg. Phrase Grounding"],
        "Avg. Perplexity": row1["Avg. Perplexity"],
        "PPL Difference": row1["PPL Difference"],
        "Avg. AMR Length": row1["Avg. AMR Length"],
        "Avg. Object Relations": row1["Avg. Object Relations"]
    }

    # Add the new row to the list
    new_rows.append(new_row)

# strip "'" from image and caption columns
# for row in new_rows:
#     row["image"] = [x.strip("'") for x in row["image"]]
#     row["caption"] = [x.strip("'") for x in row["caption"]]
# Create a new DataFrame from the list of new rows
output_df = pd.DataFrame(new_rows)

# Write the output DataFrame to a new CSV file
output_df.to_csv("zeno_code/metadata_group_view.csv", index=False)
