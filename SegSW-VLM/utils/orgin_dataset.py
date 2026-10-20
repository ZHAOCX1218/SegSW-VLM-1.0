from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN,DEFAULT_IMAGE_VISION_START,
DEFAULT_IMAGE_VISION_END,
DEFAULT_IMAGE_PAD,
)
import os
import random
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
import glob
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST
import re

TARGET_CLASS_NAME = "solid waste"
DEFAULT_DESCRIPTION = (
    "This is a remote sensing image containing possible solid waste disposal "
    "or dumping areas."
)

ATTRIBUTE_KEYS = (
    "waste_type",
    "scene_type",
    "shape_form",
    "color",
    "accumulation_boundary",
    "density",
)

ATTRIBUTE_VOCABS = {
    "waste_type": [
        "unknown", "none", "industrial waste", "household waste",
        "construction waste", "agricultural waste", "other waste", "mixed waste",
    ],
    "scene_type": [
        "unknown", "background", "industrial area", "construction area", "roadside",
        "waste dumping, landfill area", "forest", "residential area", "grassland",
        "near river channel/port", "public facilities", "farmland", "bare land",
        "mining area", "beach", "lake", "urban built-up area", "port",
        "suburban open space", "other",
    ],
    "shape_form": ["unknown", "none", "pile", "mixed", "blocky", "fragmented", "strip", "sheet"],
    "color": [
        "unknown", "none", "grayish white", "white", "grayish black", "brown",
        "black", "blue", "earthy yellow", "multicolored", "brownish-yellow",
        "red", "green", "yellow", "reddish-brown", "orange", "purple",
        "brownish yellow", "gray", "other", "识别失败", "none法判断",
        "无法判断", "unidentifiable",
    ],
    "accumulation_boundary": [
        "unknown", "none", "messy", "locally aggregated", "regular",
        "diffuse", "no solid waste", "none法判断",
    ],
    "density": [
        "unknown", "none", "dense", "scattered", "patchy",
        "locally aggregated", "sparse", "no solid waste",
    ],
}

ATTRIBUTE_CN_PATTERNS = {
    "waste_type": [
        ("无固废", "none"), ("未见明显固废", "none"), ("工业垃圾", "industrial waste"),
        ("生活垃圾", "household waste"), ("建筑垃圾", "construction waste"),
        ("农业垃圾", "agricultural waste"), ("其他垃圾", "other waste"),
    ],
    "scene_type": [
        ("无固废背景", "background"), ("工业区", "industrial area"),
        ("施工区", "construction area"), ("道路两侧", "roadside"),
        ("垃圾填埋", "waste dumping, landfill area"), ("垃圾堆放", "waste dumping, landfill area"),
        ("森林", "forest"), ("居民区", "residential area"), ("草地", "grassland"),
        ("河道", "near river channel/port"), ("港口", "near river channel/port"),
        ("公共设施", "public facilities"), ("农田", "farmland"),
        ("裸地", "bare land"), ("矿区", "mining area"), ("海滩", "beach"),
        ("湖", "lake"), ("城市背景", "urban built-up area"),
    ],
    "shape_form": [
        ("无固废", "none"), ("堆状", "pile"), ("混合", "mixed"),
        ("块状", "blocky"), ("碎状", "fragmented"), ("条带", "strip"),
        ("片状", "sheet"),
    ],
    "color": [
        ("无固废", "none"), ("灰白色", "grayish white"), ("白色", "white"),
        ("灰黑色", "grayish black"), ("棕色", "brown"), ("黑色", "black"),
        ("蓝色", "blue"), ("土黄色", "earthy yellow"), ("多色", "multicolored"),
        ("棕黄色", "brownish-yellow"), ("红色", "red"), ("绿色", "green"),
        ("黄色", "yellow"), ("红棕色", "reddish-brown"), ("橙色", "orange"),
        ("紫色", "purple"),
    ],
    "accumulation_boundary": [
        ("无固废", "none"), ("杂乱", "messy"), ("局部聚集", "locally aggregated"),
        ("规则", "regular"), ("扩散", "diffuse"),
    ],
    "density": [
        ("无固废", "none"), ("密集", "dense"), ("零散", "scattered"),
        ("斑块", "patchy"), ("局部聚集", "locally aggregated"), ("稀疏", "sparse"),
    ],
}


def _norm_attr_value(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def _vocab_index(key, value):
    vocab = ATTRIBUTE_VOCABS[key]
    value = _norm_attr_value(value)
    if not value:
        return 0
    for i, item in enumerate(vocab):
        if value == item.lower():
            return i
    for i, item in enumerate(vocab):
        if item.lower() in value or value in item.lower():
            return i
    return 0


def _infer_attribute_from_text(key, text):
    text = text or ""
    lowered = text.lower()
    for cn, canonical in ATTRIBUTE_CN_PATTERNS.get(key, []):
        if cn in text:
            return canonical
    for item in ATTRIBUTE_VOCABS[key]:
        if item != "unknown" and item.lower() in lowered:
            return item
    return "unknown"


def _pair_by_stem(base_image_dir, split):
    img_dir = os.path.join(base_image_dir, split, "img")
    lbl_dir = os.path.join(base_image_dir, split, "lbl")
    json_dir = os.path.join(base_image_dir, split, "json")
    image_exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    images = []
    for ext in image_exts:
        images.extend(glob.glob(os.path.join(img_dir, ext)))
    images = sorted(images)

    paired_images = []
    masks = []
    texts = []
    for image_path in images:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            candidate = os.path.join(lbl_dir, stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        json_path = os.path.join(json_dir, stem + ".json")
        if mask_path is not None:
            paired_images.append(image_path)
            masks.append(mask_path)
            texts.append(json_path)
    return texts, paired_images, masks


def _load_json_data(text_path):
    if not os.path.exists(text_path):
        return {}
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_description(text_path):
    data = _load_json_data(text_path)
    if not data:
        return DEFAULT_DESCRIPTION

    for key in ("img_description_en", "img_description_cn", "img_description_eg", "img_description", "description", "text", "caption"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return DEFAULT_DESCRIPTION


def _is_negative_json_data(data):
    if not data:
        return False
    values = " ".join(
        str(value).lower()
        for value in data.values()
        if isinstance(value, (str, int, float))
    )
    negative_phrases = (
        "no solid waste",
        "without solid waste",
        "no obvious solid waste",
        "negative sample",
        "未见固废",
        "未见明显固废",
        "无固废",
        "负样本",
        "全部视为背景",
    )
    if any(phrase in values for phrase in negative_phrases):
        return True
    waste_type = _norm_attr_value(data.get("waste_type"))
    return waste_type in {"none", "no solid waste", "background"}


def _is_negative_sample(text_path):
    return _is_negative_json_data(_load_json_data(text_path))


def _load_attribute_ids(text_path):
    data = _load_json_data(text_path)
    desc = ""
    for key in ("img_description_en", "img_description_cn", "img_description_eg", "img_description", "description", "text", "caption"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            desc += " " + value.strip()

    if _is_negative_json_data(data):
        negative_values = {
            "waste_type": "none",
            "scene_type": "background",
            "shape_form": "none",
            "color": "none",
            "accumulation_boundary": "none",
            "density": "none",
        }
        return torch.LongTensor(
            [_vocab_index(key, negative_values[key]) for key in ATTRIBUTE_KEYS]
        )

    attr_ids = []
    for key in ATTRIBUTE_KEYS:
        value = data.get(key)
        if value is None or not str(value).strip():
            value = _infer_attribute_from_text(key, desc)
        attr_ids.append(_vocab_index(key, value))
    return torch.LongTensor(attr_ids)


def init_gswseg4mm(base_image_dir):
    return _pair_by_stem(base_image_dir, "train")


def init_segsw_vlm(base_image_dir):
    return init_gswseg4mm(base_image_dir)

class orginDataset(torch.utils.data.Dataset):
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
        sem_seg_data="gswseg4mm",
        patch_sampling=True,
        positive_crop_prob=0.7,
        negative_sample_prob=0.5,
        crop_sizes=(256, 384, 512),
        min_foreground_pixels=32,
        crop_attempts=20,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.patch_sampling = patch_sampling
        self.positive_crop_prob = positive_crop_prob
        self.negative_sample_prob = float(negative_sample_prob)
        self.crop_sizes = tuple(int(size) for size in crop_sizes if int(size) > 0)
        self.min_foreground_pixels = min_foreground_pixels
        self.crop_attempts = crop_attempts

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.data2list = {}
        self.sem_seg_datas = sem_seg_data
        ds =self.sem_seg_datas
        text, images, labels = eval("init_{}".format(ds))(base_image_dir)
        self.data2list[ds] = (text, images, labels)
        self.length=len(images)
        self.negative_indices = [
            index for index, path in enumerate(text) if _is_negative_sample(path)
        ]
        negative_index_set = set(self.negative_indices)
        self.positive_indices = [
            index for index in range(len(images)) if index not in negative_index_set
        ]
        print('len(images):',len(images))
        print(
            "train positive/negative pools: {}/{}; negative_sample_prob={:.3f}".format(
                len(self.positive_indices),
                len(self.negative_indices),
                self.negative_sample_prob,
            )
        )
    def __len__(self):
        return self.samples_per_epoch

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

    def _crop_window(self, height, width, crop_size, center_y=None, center_x=None):
        crop_h = min(crop_size, height)
        crop_w = min(crop_size, width)
        if center_y is None or center_x is None:
            y0 = random.randint(0, max(height - crop_h, 0))
            x0 = random.randint(0, max(width - crop_w, 0))
        else:
            y0 = int(center_y) - crop_h // 2
            x0 = int(center_x) - crop_w // 2
            y0 = min(max(y0, 0), max(height - crop_h, 0))
            x0 = min(max(x0, 0), max(width - crop_w, 0))
        return y0, x0, y0 + crop_h, x0 + crop_w

    def _sample_patch(self, image, mask):
        if not self.patch_sampling or not self.crop_sizes:
            return image, mask

        height, width = mask.shape[:2]
        crop_size = random.choice(self.crop_sizes)
        foreground = np.argwhere(mask > 0)
        want_positive = len(foreground) > 0 and random.random() < self.positive_crop_prob

        if want_positive:
            for _ in range(self.crop_attempts):
                center_y, center_x = foreground[random.randrange(len(foreground))]
                y0, x0, y1, x1 = self._crop_window(height, width, crop_size, center_y, center_x)
                patch_mask = mask[y0:y1, x0:x1]
                if int(patch_mask.sum()) >= self.min_foreground_pixels:
                    return image[y0:y1, x0:x1], patch_mask

        for _ in range(self.crop_attempts):
            y0, x0, y1, x1 = self._crop_window(height, width, crop_size)
            patch_mask = mask[y0:y1, x0:x1]
            if int(patch_mask.sum()) == 0:
                return image[y0:y1, x0:x1], patch_mask

        if len(foreground) > 0:
            center_y, center_x = foreground[random.randrange(len(foreground))]
            y0, x0, y1, x1 = self._crop_window(height, width, crop_size, center_y, center_x)
        else:
            y0, x0, y1, x1 = self._crop_window(height, width, crop_size)
        return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    def __getitem__(self, idx):
        ds = self.sem_seg_datas
        texts,images, labels= self.data2list[ds]
        # print("refer: ", len(image))
        if (
            self.negative_indices
            and self.positive_indices
            and random.random() < self.negative_sample_prob
        ):
            idx = random.choice(self.negative_indices)
        elif self.positive_indices:
            idx = random.choice(self.positive_indices)
        elif self.negative_indices:
            idx = random.choice(self.negative_indices)
        else:
            idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        label_path = labels[idx]
        text_path=texts[idx]
        text = _load_description(text_path)
        attribute_ids = _load_attribute_ids(text_path)
        sampled_classes=text
        questions = []
        answers = []
        text = text.strip()
        assert len(text.split("||")) == 1
        question_template = random.choice(self.short_question_list)
        questions.append(question_template.format(text_name=text,class_name=TARGET_CLASS_NAME))
        answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = Image.open(label_path)
        masks = np.array(mask)
        if masks.ndim == 3:
            masks = masks.max(axis=2)
        masks[masks != 0] = 1
        image, masks = self._sample_patch(image, masks.astype(np.uint8))
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks =  masks[np.newaxis,:,:]
        # masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            attribute_ids,
        )
