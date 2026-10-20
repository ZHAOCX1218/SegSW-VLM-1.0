import glob
import os
import random
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
import json
import re

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .orgin_dataset import _load_attribute_ids, orginDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)

TARGET_CLASS_NAME = "solid waste"
DEFAULT_DESCRIPTION = (
    "This is a remote sensing image containing possible solid waste disposal "
    "or dumping areas."
)
def save_mask(pred_masks, save_path_dir, image_path):
    import os
    import cv2
    import numpy as np
    import torch

    os.makedirs(save_path_dir, exist_ok=True)

    image_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_np is None:
        print("Failed to read image:", image_path)
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.detach().cpu()

    for i, pred_mask in enumerate(pred_masks):
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.detach().cpu().numpy()

        pred_mask = np.squeeze(pred_mask)
        pred_mask = (pred_mask > 0).astype(np.uint8)

        h, w = image_np.shape[:2]
        if pred_mask.shape[:2] != (h, w):
            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 1) binary mask: background black, predicted solid waste white
        binary_mask = (pred_mask * 255).astype(np.uint8)
        mask_path = os.path.join(save_path_dir, f"{base_name}_mask_{i}.png")
        cv2.imwrite(mask_path, binary_mask)

        # 2) red overlay on original image
        overlay = image_np.copy()
        red = np.zeros_like(image_np)
        red[:, :, 2] = 255  # OpenCV BGR red

        fg = pred_mask > 0
        overlay[fg] = (
            0.5 * image_np[fg].astype(np.float32)
            + 0.5 * red[fg].astype(np.float32)
        ).astype(np.uint8)

        overlay_path = os.path.join(save_path_dir, f"{base_name}_masked_img_{i}.png")
        cv2.imwrite(overlay_path, overlay)



def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    attribute_ids_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    # local_text=[]
    for sample in batch:
        if len(sample) == 11:
            (
                image_path,
                images,
                images_clip,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_classes,
                attribute_ids,
                inference,
            ) = sample
        elif len(sample) == 10:
            (
                image_path,
                images,
                images_clip,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_classes,
                inference,
            ) = sample
            attribute_ids = torch.zeros(6, dtype=torch.long)
        else:
            raise ValueError(f"Unexpected dataset sample length: {len(sample)}")
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        attribute_ids_list.append(attribute_ids.long())
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
            #<im_start>+<image>+<im_end>
   
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
   
    #填充序列
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
   
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep
            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
       
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255
        # print('input_ids.shape[1]',input_ids.shape[1])
        # print('input_ids.shape[0]',input_ids.shape[0])
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
     
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "attribute_ids": torch.stack(attribute_ids_list, dim=0),
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="gswseg4mm",
        sample_rate=[9, 3, 3, 1],
        seg_data="FIT|LoveDA",
        explanatory=0.1,
        patch_sampling=True,
        positive_crop_prob=0.7,
        negative_sample_prob=0.5,
        crop_sizes=(256, 384, 512),
        min_foreground_pixels=32,
        crop_attempts=20,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        self.datasets = dataset.split("||")
        sample_rate = np.array(sample_rate, dtype=np.float32)
        if len(sample_rate) != len(self.datasets):
            sample_rate = np.ones(len(self.datasets), dtype=np.float32)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.all_datasets = []
        for dataset in self.datasets:
            if dataset in ("gswseg4mm", "segsw_vlm"):
                self.all_datasets.append(
                    orginDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        seg_data,
                        patch_sampling=patch_sampling,
                        positive_crop_prob=positive_crop_prob,
                        negative_sample_prob=negative_sample_prob,
                        crop_sizes=crop_sizes,
                        min_foreground_pixels=min_foreground_pixels,
                        crop_attempts=crop_attempts,
                    )
                )
           
    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        
        return *data[0], inference

def _find_image_for_mask(mask_path):
    parent = os.path.dirname(os.path.dirname(mask_path))
    img_dir = os.path.join(parent, "img")
    stem = os.path.splitext(os.path.basename(mask_path))[0]
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        candidate = os.path.join(img_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_description_for_image(image_path):
    parent = os.path.dirname(os.path.dirname(image_path))
    json_dir = os.path.join(parent, "json")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(json_dir, stem + ".json")
    if not os.path.exists(json_path):
        return DEFAULT_DESCRIPTION
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return DEFAULT_DESCRIPTION
    for key in ("img_description_en", "img_description_cn", "img_description_eg", "img_description", "description", "text", "caption"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return DEFAULT_DESCRIPTION


def _load_attributes_for_image(image_path):
    parent = os.path.dirname(os.path.dirname(image_path))
    json_dir = os.path.join(parent, "json")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return _load_attribute_ids(os.path.join(json_dir, stem + ".json"))


def load_data(base_dir, split="val", region=None):
    split_dir = os.path.join(base_dir, split)
    if region:
        mask_roots = [os.path.join(split_dir, region, "lbl")]
    else:
        mask_roots = [os.path.join(split_dir, "lbl")]
        mask_roots.extend(glob.glob(os.path.join(split_dir, "*", "lbl")))

    masks = []
    for mask_root in mask_roots:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
            masks.extend(glob.glob(os.path.join(mask_root, ext)))
    masks = sorted(set(masks))

    images = []
    paired_masks = []
    for mask_path in masks:
        image_path = _find_image_for_mask(mask_path)
        if image_path is not None:
            images.append(image_path)
            paired_masks.append(mask_path)
    return images, paired_masks
class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        if "ViT-L-14" in vision_tower:
            vision_tower="openai/clip-vit-large-patch14"
        self.base_image_dir = base_image_dir
        parts = str(val_dataset).split("/", 1)
        split = parts[0] if parts[0] else "val"
        region = parts[1] if len(parts) > 1 else None
        self.images,self.mask=load_data(self.base_image_dir, split=split, region=region)
        print('val:',len(self.images))
        ds="dataset"
        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        image_path=self.images[idx]
        # text_path=self.texts[idx]
        # f = open(text_path, encoding = 'utf-8')
        # path=f.read()
        text = _load_description_for_image(image_path)
        attribute_ids = _load_attributes_for_image(image_path)
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        
        conv.append_message(
                conv.roles[0],
                DEFAULT_IMAGE_TOKEN
                + "\n {} Can you segment the solid waste in this image?.".format(text),
                )
        conv.append_message(conv.roles[1], "[SEG].")
        conversations.append(conv.get_prompt())
        i += 1

        # preprocess image for clip
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

       
        masks_path=self.mask[idx]
        mask = Image.open(masks_path)
        masks = np.array(mask)
        if masks.ndim == 3:
            masks = masks.max(axis=2)
        masks[masks!=0]=1
        # masks=masks.max(axis=2)
        masks =  masks[np.newaxis,:,:]
        # masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            attribute_ids,
            inference,
        )
