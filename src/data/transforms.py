from __future__ import annotations
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

# CLIP default image size is often 224 for ViT-B/32
DEFAULT_IMAGE_SIZE = 224

# CLIP normalization stats
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

class ResizeAndPad:
    def __init__(self, size: int, fill: int = 0) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size

        if width == 0 or height == 0:
            raise ValueError("Input image has invalid size.")

        scale = self.size / max(width, height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        img = img.resize((new_width, new_height), resample=Image.BICUBIC)

        pad_w = self.size - new_width
        pad_h = self.size - new_height

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        img = F.pad(img, padding=[left, top, right, bottom], fill=self.fill)

        return img


def build_train_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> T.Compose:
    return T.Compose([ResizeAndPad(size=image_size, fill=0),
                      T.RandomHorizontalFlip(p=0.5),
                      T.ToTensor(),
                      T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])


def build_eval_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> T.Compose:
    return T.Compose([ResizeAndPad(size=image_size, fill=0),
                      T.ToTensor(),
                      T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])