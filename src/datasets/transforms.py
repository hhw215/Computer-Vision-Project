from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    """Build ViT-friendly transforms with ImageNet normalization."""
    resize_size = int((256 / 224) * image_size)

    if train:
        return transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
