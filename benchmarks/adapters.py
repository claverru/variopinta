from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from common import MEAN, SEED, STD


class Adapter:
    def __init__(self, backend: str) -> None:
        self.backend = backend

    def native_inputs(self, images: list[np.ndarray]) -> list[Any]:
        if self.backend == "torchvision":
            import torch

            return [torch.from_numpy(x).permute(2, 0, 1).contiguous() for x in images]
        return images

    @staticmethod
    def _materialize(value: Any) -> Any:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return value.contiguous()
        except ImportError:
            pass
        return np.ascontiguousarray(value)

    def build_micro(self, name: str, size: int) -> Callable[[Any], Any]:
        out = max(32, size * 3 // 4)
        if self.backend == "torchvision":
            return self._torch_transform(name, out, micro=True)
        if self.backend in {"albumentations", "albumentationsx"}:
            return self._albu_transform(name, out, micro=True)
        return self._rust_transform(name, out, micro=True)

    def build_pipeline(
        self, size: int = 512, name: str = "classic", *, to_torch: bool = False
    ) -> Callable[[Any], Any]:
        crop = max(224, size * 7 // 8)
        if self.backend == "torchvision":
            return self._torch_pipeline(crop, name)
        if self.backend in {"albumentations", "albumentationsx"}:
            return self._albu_pipeline(crop, name, to_torch=to_torch)
        return self._rust_pipeline(crop, name, to_torch=to_torch)

    def build_antialiased_resize(self, size: int) -> Callable[[Any], Any] | None:
        out = max(32, size * 3 // 4)
        if self.backend == "torchvision":
            from torchvision.transforms import InterpolationMode, v2

            transform = v2.Resize(
                (out, out), interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            return lambda image: self._materialize(transform(image))
        if self.backend == "rust":
            import variopinta as R

            transform = R.Compose([R.Resize(out, out, antialias=True)], seed=SEED).compile()
            return lambda image: self._materialize(transform(image))
        return None

    @staticmethod
    def _seed_albu(transform: Any) -> Any:
        if hasattr(transform, "set_random_seed"):
            transform.set_random_seed(SEED)
        return transform

    def _torch_transform(self, name: str, out: int, micro: bool) -> Callable[[Any], Any]:
        import torch
        from torchvision.transforms import InterpolationMode, v2

        transforms: dict[str, Any] = {
            "Resize": v2.Resize(
                (out, out), interpolation=InterpolationMode.BILINEAR, antialias=False
            ),
            "RandomCrop": v2.RandomCrop((out, out)),
            "CenterCrop": v2.CenterCrop((out, out)),
            "HorizontalFlip": v2.RandomHorizontalFlip(p=1.0 if micro else 0.5),
            "VerticalFlip": v2.RandomVerticalFlip(p=1.0 if micro else 0.5),
            "ColorJitter": v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            "Affine": v2.RandomAffine(
                degrees=10.0, interpolation=InterpolationMode.BILINEAR, fill=0
            ),
            "GaussianBlur": v2.GaussianBlur(kernel_size=5, sigma=(1.1, 1.1)),
            "Grayscale": v2.Grayscale(num_output_channels=3),
            "Invert": v2.RandomInvert(p=1.0),
            "Solarize": v2.RandomSolarize(threshold=128, p=1.0),
            "Posterize": v2.RandomPosterize(bits=4, p=1.0),
            "Normalize": v2.Compose(
                [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
            ),
        }
        torch.manual_seed(SEED)
        transform = transforms[name]
        return lambda image: self._materialize(transform(image))

    def _torch_pipeline(self, crop: int, name: str) -> Callable[[Any], Any]:
        import torch
        from torchvision.transforms import InterpolationMode, v2

        torch.manual_seed(SEED)
        classic = [
            v2.RandomCrop((crop, crop)),
            v2.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=False),
            v2.RandomHorizontalFlip(0.5),
            v2.ColorJitter(0.2, 0.2, 0.2, 0.0),
            v2.RandomAffine(10.0, interpolation=InterpolationMode.BILINEAR, fill=0),
            v2.GaussianBlur(5, (1.1, 1.1)),
        ]
        pipelines = {
            "classic": classic,
            "extended": [
                v2.RandomCrop((crop, crop)),
                v2.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=False),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.2),
                v2.ColorJitter(0.2, 0.2, 0.2, 0.0),
                v2.RandomAffine(10.0, interpolation=InterpolationMode.BILINEAR, fill=0),
                v2.GaussianBlur(5, (1.1, 1.1)),
                v2.RandomGrayscale(0.1),
                v2.RandomSolarize(128, 0.2),
                v2.RandomPosterize(4, 0.2),
            ],
            "pixel_policy": [
                v2.CenterCrop((crop, crop)),
                v2.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=False),
                v2.RandomGrayscale(0.2),
                v2.RandomInvert(0.1),
                v2.RandomSolarize(128, 0.2),
                v2.RandomPosterize(4, 0.2),
            ],
        }
        transform = v2.Compose(
            [*pipelines[name], v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
        )
        return lambda image: self._materialize(transform(image))

    def _albu_transform(self, name: str, out: int, micro: bool) -> Callable[[np.ndarray], Any]:
        import albumentations as A
        import cv2

        if self.backend == "albumentationsx":
            jitter = A.ColorJitter(
                brightness_range=(0.8, 1.2),
                contrast_range=(0.8, 1.2),
                saturation_range=(0.8, 1.2),
                hue_range=(0.0, 0.0),
                p=1,
            )
            affine = A.Affine(
                scale=(1.0, 1.0),
                translate_percent=(0.0, 0.0),
                rotate=(-10.0, 10.0),
                shear=(0.0, 0.0),
                interpolation=cv2.INTER_LINEAR,
                fill=0,
                p=1,
            )
            blur = A.GaussianBlur(blur_range=(5, 5), sigma_range=(1.1, 1.1), p=1)
        else:
            jitter = A.ColorJitter(0.2, 0.2, 0.2, 0.0, p=1)
            affine = A.Affine(
                scale=1.0,
                translate_percent=0.0,
                rotate=(-10.0, 10.0),
                shear=0.0,
                interpolation=cv2.INTER_LINEAR,
                fill=0,
                p=1,
            )
            blur = A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(1.1, 1.1), p=1)

        transforms: dict[str, Any] = {
            "Resize": A.Resize(out, out, interpolation=cv2.INTER_LINEAR, p=1),
            "RandomCrop": A.RandomCrop(out, out, p=1),
            "CenterCrop": A.CenterCrop(out, out, p=1),
            "HorizontalFlip": A.HorizontalFlip(p=1.0 if micro else 0.5),
            "VerticalFlip": A.VerticalFlip(p=1.0 if micro else 0.5),
            "ColorJitter": jitter,
            "Affine": affine,
            "GaussianBlur": blur,
            "Grayscale": A.ToGray(num_output_channels=3, method="weighted_average", p=1),
            "Invert": A.InvertImg(p=1),
            "Solarize": A.Solarize(threshold_range=(128 / 255, 128 / 255), p=1),
            "Posterize": A.Posterize(
                num_bits=(4, 4) if self.backend == "albumentationsx" else 4, p=1
            ),
            "Normalize": A.Compose([A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0, p=1)]),
        }
        transform = self._seed_albu(
            A.Compose([transforms[name]]) if name != "Normalize" else transforms[name]
        )
        return lambda image: self._materialize(transform(image=image)["image"])

    def _albu_pipeline(
        self, crop: int, name: str, *, to_torch: bool = False
    ) -> Callable[[np.ndarray], Any]:
        import albumentations as A
        import cv2

        if self.backend == "albumentationsx":
            jitter = A.ColorJitter(
                brightness_range=(0.8, 1.2),
                contrast_range=(0.8, 1.2),
                saturation_range=(0.8, 1.2),
                hue_range=(0.0, 0.0),
                p=1,
            )
            affine = A.Affine(
                scale=(1.0, 1.0),
                translate_percent=(0.0, 0.0),
                rotate=(-10.0, 10.0),
                shear=(0.0, 0.0),
                interpolation=cv2.INTER_LINEAR,
                fill=0,
                p=1,
            )
            blur = A.GaussianBlur(blur_range=(5, 5), sigma_range=(1.1, 1.1), p=1)
        else:
            jitter = A.ColorJitter(0.2, 0.2, 0.2, 0.0, p=1)
            affine = A.Affine(
                scale=1.0,
                translate_percent=0.0,
                rotate=(-10.0, 10.0),
                shear=0.0,
                interpolation=cv2.INTER_LINEAR,
                fill=0,
                p=1,
            )
            blur = A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(1.1, 1.1), p=1)

        posterize_bits = (4, 4) if self.backend == "albumentationsx" else 4
        classic = [
            A.RandomCrop(crop, crop, p=1),
            A.Resize(224, 224, interpolation=cv2.INTER_LINEAR, p=1),
            A.HorizontalFlip(p=0.5),
            jitter,
            affine,
            blur,
        ]
        pipelines = {
            "classic": classic,
            "extended": [
                A.RandomCrop(crop, crop, p=1),
                A.Resize(224, 224, interpolation=cv2.INTER_LINEAR, p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                jitter,
                affine,
                blur,
                A.ToGray(num_output_channels=3, method="weighted_average", p=0.1),
                A.Solarize(threshold_range=(128 / 255, 128 / 255), p=0.2),
                A.Posterize(num_bits=posterize_bits, p=0.2),
            ],
            "pixel_policy": [
                A.CenterCrop(crop, crop, p=1),
                A.Resize(224, 224, interpolation=cv2.INTER_LINEAR, p=1),
                A.ToGray(num_output_channels=3, method="weighted_average", p=0.2),
                A.InvertImg(p=0.1),
                A.Solarize(threshold_range=(128 / 255, 128 / 255), p=0.2),
                A.Posterize(num_bits=posterize_bits, p=0.2),
            ],
        }
        terminal: list[Any] = [A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0, p=1)]
        if to_torch:
            from albumentations.pytorch import ToTensorV2

            terminal.append(ToTensorV2())
        transform = A.Compose([*pipelines[name], *terminal])
        self._seed_albu(transform)
        return lambda image: self._materialize(transform(image=image)["image"])

    def _rust_transform(self, name: str, out: int, micro: bool) -> Callable[[np.ndarray], Any]:
        import variopinta as R

        transforms: dict[str, Any] = {
            "Resize": R.Resize(out, out),
            "RandomCrop": R.RandomCrop(out, out),
            "CenterCrop": R.CenterCrop(out, out),
            "HorizontalFlip": R.HorizontalFlip(1.0 if micro else 0.5),
            "VerticalFlip": R.VerticalFlip(1.0 if micro else 0.5),
            "ColorJitter": R.ColorJitter(0.2, 0.2, 0.2),
            "Affine": R.Affine(10.0),
            "GaussianBlur": R.GaussianBlur(5, 1.1),
            "Grayscale": R.Grayscale(),
            "Invert": R.Invert(),
            "Solarize": R.Solarize(128),
            "Posterize": R.Posterize(4),
            "Normalize": R.Normalize(MEAN, STD),
        }
        transform = R.Compose([transforms[name]], seed=SEED).compile()
        return lambda image: self._materialize(transform(image))

    def _rust_pipeline(
        self, crop: int, name: str, *, to_torch: bool = False
    ) -> Callable[[np.ndarray], Any]:
        import variopinta as R

        classic = [
            R.RandomCrop(crop, crop),
            R.Resize(224, 224),
            R.HorizontalFlip(0.5),
            R.ColorJitter(0.2, 0.2, 0.2),
            R.Affine(10.0),
            R.GaussianBlur(5, 1.1),
        ]
        pipelines = {
            "classic": classic,
            "extended": [
                R.RandomCrop(crop, crop),
                R.Resize(224, 224),
                R.HorizontalFlip(0.5),
                R.VerticalFlip(0.2),
                R.ColorJitter(0.2, 0.2, 0.2),
                R.Affine(10.0),
                R.GaussianBlur(5, 1.1),
                R.Grayscale(0.1),
                R.Solarize(128, 0.2),
                R.Posterize(4, 0.2),
            ],
            "pixel_policy": [
                R.CenterCrop(crop, crop),
                R.Resize(224, 224),
                R.Grayscale(0.2),
                R.Invert(0.1),
                R.Solarize(128, 0.2),
                R.Posterize(4, 0.2),
            ],
        }
        terminal: list[Any] = [R.Normalize(MEAN, STD)]
        if to_torch:
            terminal.append(R.ToTorch())
        transform = R.Compose([*pipelines[name], *terminal], seed=SEED).compile()

        def apply(image: np.ndarray) -> Any:
            return self._materialize(transform(image))

        apply.native = transform  # type: ignore[attr-defined]
        return apply
