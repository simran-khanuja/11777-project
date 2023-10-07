import json

def extract_caption_and_amr(data):
    results = []
    for entry in data:
        captions = entry.get('caption_pair', [])
        amrs = entry.get('amr', [])
        for caption, amr in zip(captions, amrs):
            results.append(f"Caption: {caption}\n{amr}\n")
    return "\n".join(results)

# Read JSON data from a file
with open('tree_dump.json', 'r') as infile:
    data = json.load(infile)

formatted_data = extract_caption_and_amr(data)

# Save the extracted data to a text file
with open('formatted_data.txt', 'w') as outfile:
    outfile.write(formatted_data)

print("Data saved to formatted_data.txt")

