import json
import nltk
from nltk import Tree

# Load the input data
with open("tree_dump.json", "r") as file:
    data = json.load(file)

# Define function to extract POS words from a tree
def extract_words(tree, pos_tags):
    return [word for word, pos in tree.pos() if pos in pos_tags]

# Initialize dictionaries to store extracted words
nouns = {}
verbs = {}
adjectives = {}
count=0

# Process each entry
for entry in data:
    for i, constituency in enumerate(entry["constituency"]):
        # Parse the constituency string into a tree
        tree = Tree.fromstring(constituency)

        # Extract and store nouns, verbs, and adjectives
        nouns[count] = ', '.join(extract_words(tree, ["NN", "NNS", "NNP", "NNPS"]))
        verbs[count] = ', '.join(extract_words(tree, ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]))
        adjectives[count] = ', '.join(extract_words(tree, ["JJ", "JJR", "JJS"]))

        count+=1

# Write extracted words into separate JSON files
with open("nouns.json", "w") as file:
    json.dump(nouns, file, indent=4)

with open("verbs.json", "w") as file:
    json.dump(verbs, file, indent=4)

with open("adjectives.json", "w") as file:
    json.dump(adjectives, file, indent=4)
