import argparse
import os

import numpy as np
import json
import torch
import torchvision
from PIL import Image
import litellm

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram, ram_plus
from ram import inference_ram
import torchvision.transforms as TS


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--ram_version", type=str, required=True, help="ram version, like: ram or ram_plus")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    split = args.split
    openai_key = args.openai_key
    openai_proxy = args.openai_proxy
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device
    
    
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])
    
    # load model
    if args.ram_version == "ram":
        ram_model = ram(pretrained=ram_checkpoint,
                                            image_size=384,
                                            vit='swin_l')
    elif args.ram_version == "ram_plus":
        ram_model = ram_plus(pretrained=ram_checkpoint,
                                        image_size=384,
                                        vit='swin_l')
    else:
        raise ValueError("ram_version can only be ram or ram_plus.")

    ram_model.eval()

    ram_model = ram_model.to(device)
    raw_image = image_pil.resize(
                    (384, 384))
    raw_image  = transform(raw_image).unsqueeze(0).to(device)

    res = inference_ram(raw_image , ram_model)

    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    tags=res[0].replace(' |', ',')

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model, image, tags, box_threshold, text_threshold, device=device
    )

    if use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]

    innn = 0
    crop_information_data_list = []
    os.makedirs(f'{output_dir}/bbox', exist_ok=True)
    for box, label in zip(boxes_filt, pred_phrases):
        sx, sy, ex, ey = map(int, box)
        sub_image = image[sy:ey, sx:ex]
        bbox_list = [sx, sy, ex, ey]
        output_path = f'{output_dir}/bbox/{innn:05}.jpg' 
        innn += 1
        sub_image_bgr = cv2.cvtColor(sub_image, cv2.COLOR_RGB2BGR) 
        cv2.imwrite(output_path, sub_image_bgr)
        label = label.split('(')[0]
        new_data = {
            'output_path': output_path,
            'label': label,
            'bbox_list': bbox_list,
            'img_H': H,
            'img_W': W
        }
        crop_information_data_list.append(new_data)

    output_json = f"{output_dir}/bbox.json"

    with open(output_json, 'w') as json_file:
        json.dump(crop_information_data_list, json_file, indent=4)
