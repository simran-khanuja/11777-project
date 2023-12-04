#%%
import numpy as np
import matplotlib.pyplot as plt
import gc
import torch
import requests
from PIL import Image

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import SamModel, SamProcessor
# from diffusers import AutoPipelineForInpainting
# from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionInpaintPipeline
from datasets import load_dataset
import spacy
import pickle 
from datasets import Dataset
from tqdm import tqdm
# Helper functions
#%%
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[1] == 1 and scores.shape[-1] != 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))
    mask = masks.cpu().detach()
    axes.imshow(np.array(raw_image))
    show_mask(mask, axes)
    axes.title.set_text(f"Mask Score: {scores.item():.3f}")
    axes.axis("off")
    plt.show()

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()
###############################################################
#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device).eval()
# Get segment anything model

sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device).eval()
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# Edit the objects to augmented noun
# ------Stable Diffusion

pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16,).to(device)

# if one wants to disable `tqdm`
pipe.set_progress_bar_config(disable=True)

#%%
def get_adj_phrases(nlp, caption):

    doc = nlp(caption)
    texts_chunks = [chunk for chunk in doc.noun_chunks]
    return texts_chunks


## Now, given image, orginal caption, augmented caption, can I get the edited image?

# Given:  image, orginal caption, augmented caption
# All pipelines are assumed global for simplicity
# The pipeline can work for 2 & 3 nouns & adjectives or more

def get_edited_image(nlp, image, original_caption, augmented_caption, seed: torch.Generator, n_steps: int=100):
    
    orginal_caption_adjective_nouns = get_adj_phrases(nlp, original_caption)
        
    augmented_caption_adjective_nouns = get_adj_phrases(nlp, augmented_caption)

    cap_str = [str(noun_adj) for noun_adj in orginal_caption_adjective_nouns]
    aug_cap_str = [str(noun_adj) for noun_adj in augmented_caption_adjective_nouns]
    # Iterate through all nouns

    # Save images and iteratvely use previous edited images
    images_to_paint = [image]
    for i in range(len(cap_str)):
        # print(f"Editing {cap_str[i]} to {aug_cap_str[i]}")
        def get_object_box(images, texts):
            texts = [texts]
            det_inputs = owl_processor(text=texts, images=images, return_tensors="pt").to(device)
            outputs = owl_model(**det_inputs)
            target_sizes = torch.Tensor([images.size[::-1]]).to(device)
            results = owl_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.005)
            object_box = results[0]["boxes"].to("cpu").detach().tolist()
            if len(object_box) > 1:
                ind_max = torch.argmax(results[0]["scores"])
                # Get the box that correspond to ind score
                object_box = results[0]["boxes"].to("cpu").detach().tolist()
                object_box = [object_box[ind_max]]
            return object_box

        object_box = get_object_box(images_to_paint[i], cap_str[i])
        # show image 
        if len(object_box) > 0:
            sam_input = sam_processor(images_to_paint[i], input_boxes=[object_box], return_tensors="pt").to(device)
            with torch.no_grad():
                output_mask = sam_model(**sam_input, multimask_output=False)
            object_mask = sam_processor.image_processor.post_process_masks(output_mask.pred_masks.cpu(), sam_input["original_sizes"].cpu(), sam_input["reshaped_input_sizes"].cpu())
            # show the mask
            object_scores = output_mask.iou_scores
            object_mask_final = object_mask[0][0] #>> [1, h, w]

            only_mask = Image.fromarray(object_mask_final[0].cpu().numpy())
            
            # show_masks_on_image(images_to_paint[i], object_mask_final, object_scores)
            prompt = aug_cap_str[i]
            edited_img = pipe(prompt=prompt, image=images_to_paint[i], mask_image=only_mask, negative_prompt=cap_str[i], generator=seed, num_inference_steps=n_steps).images[0]
            images_to_paint.append(edited_img)
            # show edited image
            # fig, axes = plt.subplots(1, 1, figsize=(15, 15))
            # axes.imshow(np.array(edited_img))
        elif i == 1:
            # No object detected, but this is the second noun, so we can use the original image's mask
            prompt = aug_cap_str[i]
            # print(prompt)
            object_box = get_object_box(images_to_paint[i-1], cap_str[i])
            sam_input = sam_processor(images_to_paint[i-1], input_boxes=[object_box], return_tensors="pt").to(device)
            with torch.no_grad():
                output_mask = sam_model(**sam_input, multimask_output=False)
            object_mask = sam_processor.image_processor.post_process_masks(output_mask.pred_masks.cpu(), sam_input["original_sizes"].cpu(), sam_input["reshaped_input_sizes"].cpu())
            # show the mask
            object_scores = output_mask.iou_scores
            object_mask_final = object_mask[0][0] #>> [1, h, w]

            only_mask = Image.fromarray(object_mask_final[0].cpu().numpy())
            # show_masks_on_image(images_to_paint[i], object_mask_final, object_scores)
            edited_img = pipe(prompt=prompt, image=images_to_paint[i], mask_image=only_mask, generator=seed, negative_prompt=cap_str[i], num_inference_steps=n_steps).images[0]
            # fig, axes = plt.subplots(1, 1, figsize=(15, 15))
            # axes.imshow(np.array(edited_img))
            images_to_paint.append(edited_img)
        else:
            return None, None


    return edited_img, augmented_caption
#%%
import pickle
coco_dataset = pickle.load(open("1filtered_dataset_step_2.pkl", "rb"))
nlp = spacy.load("en_core_web_lg")

# %%
seed = torch.Generator(device="cuda")
failure_cases = []
failure_count = 0
new_data = []
for idx, data in tqdm(enumerate(coco_dataset), total=len(coco_dataset), leave=False, colour="green"):
    image = data["image"]
    original_caption = data["sentences"]["raw"]
    augmented_caption = data["caption_pair"][1]
    edited_img, augmented_caption = get_edited_image(nlp, image, original_caption, augmented_caption, seed=seed, n_steps=70)
    if edited_img is None:
        print(f"failed at {idx} with {original_caption} and {augmented_caption}")
        failure_cases.append(data)
        failure_count += 1
    else:
        data["aug_image"] = edited_img
        new_data.append(data)
    if (idx+1) % 50 == 0:
        new_dataset = Dataset.from_list(new_data)
        with open("filtered_dataset_step_3.pkl", "wb") as f:
            pickle.dump(new_dataset, f)
        if failure_count > 0:
            failure_dataset = Dataset.from_list(failure_cases)
            with open("filtered_dataset_step_3_failures.pkl", "wb") as f:
                pickle.dump(failure_cases, f)
        print(f"Saved at {idx}")

new_dataset = Dataset.from_list(new_data)
with open("filtered_dataset_step_3.pkl", "wb") as f:
    pickle.dump(new_dataset, f)
if failure_count > 0:
    failure_dataset = Dataset.from_list(failure_cases)
    with open("filtered_dataset_step_3_failures.pkl", "wb") as f:
        pickle.dump(failure_cases, f)
print(f"Saved at {idx}")