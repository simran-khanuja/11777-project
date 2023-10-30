import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import pdb
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="facebook/opt-6.7b")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--image-file", type=str, required=False)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--image-aspect-ratio", type=str, default='pad')
args = parser.parse_args()

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

def group_correct(result):
    return image_correct(result) and text_correct(result)

def inference_winoground():
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    from datasets import load_dataset
    auth_token = "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"  # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
    winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]
    winoground.set_format("torch")
    def transform_wino(examples):
        examples["image_0"] = [image.convert("RGB") for image in examples["image_0"]]
        examples["image_1"] = [image.convert("RGB") for image in examples["image_1"]]
        return examples

    winoground.set_transform(transform_wino)
    device = 'cuda:0'
    from tqdm import tqdm

    llava_scores = []

    ID_Yes = 3869
    ID_yes = 4874
    ID_YES = 22483
    ID_No = 1939
    ID_no = 694 
    ID_NO = 11698

    for sample in tqdm(winoground):
        sample_image_0 = Image.fromarray(np.array(sample["image_0"]))
        sample_image_1 = Image.fromarray(np.array(sample["image_1"]))
        
        image_0, cap_0 = process_images([sample_image_0], image_processor, args), sample["caption_0"]
        # pdb.set_trace()
        
        if type(image_0) is list:
            image_0 = [image.to(model.device, dtype=torch.float16) for image in image_0]
        else:
            image_0 = image_0.to(model.device, dtype=torch.float16)
        
        image_1, cap_1 = process_images([sample_image_1], image_processor, args), sample["caption_1"]
        if type(image_1) is list:
            image_1 = [image.to(model.device, dtype=torch.float16) for image in image_1]
        else:
            image_1 = image_1.to(model.device, dtype=torch.float16)

        # Preprocess the images
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        text_inp_0 = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n{roles[0]}: Is caption: '{sample['caption_0']}' the correct description for the image?"
        text_inp_1 = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n{roles[0]}: Is caption: '{sample['caption_1']}' the correct description for the image?"
        # image_inp_0 = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {roles[0]}: Does the caption: '{sample['caption_0']}' describe the first image (A) better, or second image(B) better?"
        # image_inp_1 = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {roles[0]}: Does the caption: '{sample['caption_1']}' describe the first image (A) better, or second image(B) better?"
        
        print(text_inp_0, text_inp_1)
        # print(f"{roles[1]}: ", end="")
        def get_inputs( inp):
            conv = conv_templates[args.conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)


            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            return conv, input_ids, stop_str, keywords, stopping_criteria, streamer

        conv, input_ids, stop_str, keywords, stopping_criteria, streamer = get_inputs(text_inp_0)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_0,
                do_sample=False,
                temperature=1e-9,
                max_new_tokens=2,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                return_dict_in_generate=True,
                output_scores=True)
            
            prob = torch.softmax(outputs.scores[0].T,0)
            prob_yes = max(prob[ID_YES][0],prob[ID_Yes][0],prob[ID_yes][0])
            prob_no = max(prob[ID_no][0], prob[ID_No][0], prob[ID_NO][0])

            c0_i0 = prob_yes

        conv, input_ids, stop_str, keywords, stopping_criteria, streamer = get_inputs(text_inp_1)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_0,
                do_sample=False,
                temperature=1e-9,
                max_new_tokens=2,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                return_dict_in_generate=True,
                output_scores=True)
            
            prob = torch.softmax(outputs.scores[0].T,0)
            prob_yes = max(prob[ID_YES][0],prob[ID_Yes][0],prob[ID_yes][0])
            prob_no = max(prob[ID_no][0], prob[ID_No][0], prob[ID_NO][0])

            c1_i0 = prob_yes 

            # print(outputs)

        conv, input_ids, stop_str, keywords, stopping_criteria, streamer = get_inputs(text_inp_0)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_1,
                do_sample=False,
                temperature=1e-9,
                max_new_tokens=2,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                return_dict_in_generate=True,
                output_scores=True)

            prob = torch.softmax(outputs.scores[0].T,0)
            prob_yes = max(prob[ID_YES][0],prob[ID_Yes][0],prob[ID_yes][0])
            prob_no = max(prob[ID_no][0], prob[ID_No][0], prob[ID_NO][0])

            c0_i1 = prob_yes
            # print(outputs)

        conv, input_ids, stop_str, keywords, stopping_criteria, streamer = get_inputs(text_inp_1)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_1,
                do_sample=False,
                temperature=1e-9,
                max_new_tokens=2,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                return_dict_in_generate=True,
                output_scores=True)
            # print(outputs)

            prob = torch.softmax(outputs.scores[0].T,0)
            prob_yes = max(prob[ID_YES][0],prob[ID_Yes][0],prob[ID_yes][0])
            prob_no = max(prob[ID_no][0], prob[ID_No][0], prob[ID_NO][0])

            c1_i1 = prob_yes
            llava_scores.append({"id" : sample["id"], "c0_i0": c0_i0, "c1_i0": c1_i0, "c0_i1": c0_i1, "c1_i1": c1_i1})

    pdb.set_trace()
    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in llava_scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    num_samples = len(winoground)
    print("text score:", text_correct_count/num_samples)
    print("image score:", image_correct_count/num_samples)
    print("group score:", group_correct_count/num_samples)

    import json
    pdb.set_trace()
    # wino_llava = {str(d['id']): {k: v for k, v in d.items() if k != 'id'} for d in llava_scores}
    with open("llava_scores.json","w") as f: 
        json.dump({str(d['id']): {"c0_i0": float(d['c0_i0']), "c1_i0": float(d['c1_i0']), "c1_i1": float(d['c1_i1']), "c0_i1": float(d['c0_i1'])} for d in llava_scores},f)

    # Get the probability scores
    # im0_cap0_scores = image_score1.item()
    # im0_cap1_scores = image_score2.item()
    # im1_cap0_scores = image_score3.item()
    # im1_cap1_scores = image_score4.item()

        # llava_scores.append({"id" : sample["id"], "c0_i0": im0_cap0_scores, "c1_i0": im0_cap1_scores, "c0_i1": im1_cap0_scores, "c1_i1": im1_cap1_scores})

inference_winoground()