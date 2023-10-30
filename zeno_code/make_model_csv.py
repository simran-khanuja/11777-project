import json
import argparse
import csv
parser = argparse.ArgumentParser()
parser.add_argument('--read_path', type=str, default='/Users/simrankhanuja/Desktop/11777-project/wino_clip_base.json')
parser.add_argument('--dump_prefix', type=str, default='/Users/simrankhanuja/Desktop/11777-project/wino_clip_base')

args = parser.parse_args()

with open(args.read_path, "r") as f:
    wino_scores = json.load(f)

rows = {}
group_rows = {}
for i in range(800):
    img_text = wino_scores[str(int(i/2))]
    c0_i0 = float(img_text["c0_i0"])
    c0_i1 = float(img_text["c0_i1"])
    c1_i0 = float(img_text["c1_i0"])
    c1_i1 = float(img_text["c1_i1"])

    if i%2 == 0:
        if c1_i1 > c0_i1:
            text_retrieval = True
        else:
            text_retrieval = False
        if c1_i1 > c1_i0:
            image_retrieval = True
        else:
            image_retrieval = False
        true_similarity = c1_i1
        diff_from_false_text = c1_i1 - c0_i1
        diff_from_false_img = c1_i1 - c1_i0
    else:
        if c0_i0 > c1_i0:
            text_retrieval = True
        else:
            text_retrieval = False
        if c0_i0 > c0_i1:
            image_retrieval = True
        else:
            image_retrieval = False
        true_similarity = c0_i0
        diff_from_false_text = c0_i0 - c1_i0
        diff_from_false_img = c0_i0 - c0_i1
    rows[i] = [true_similarity, diff_from_false_text, diff_from_false_img, text_retrieval, image_retrieval]


# Write to csv file

columns = ["True Similarity", "Diff from False - Text Retrieval", "Diff from False - Image Retrieval", "Text Retrieval", "Image Retrieval"]

with open(f'{args.dump_prefix}.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(columns)
    for i in range(800):
        write.writerow(rows[i])

for row in rows:
    if row%2==0:
        group_rows[row/2] = [rows[row][3] & rows[row+1][3], rows[row][4] & rows[row+1][4]]

with open(f'{args.dump_prefix}_grp.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(["Group Text Retrieval", "Group Image Retrieval"])
    for i in range(400):
        write.writerow(group_rows[i])
    