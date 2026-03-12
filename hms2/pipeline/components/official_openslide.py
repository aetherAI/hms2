import typing as t
from dataclasses import dataclass

import cv2
import numpy as np
import openslide
from PIL import Image


@dataclass
class BoundingBox:
    left: int
    top: int
    width: int
    height: int

    def __init__(self, left: int, top: int, width: int, height: int):
        self.left = left
        self.top = top
        self.width = width
        self.height = height


@dataclass
class SlideMetadata:
    width: int
    height: int
    pixel_width_nm: t.Optional[float]
    pixel_height_nm: t.Optional[float]
    lens: t.Optional[float]


class OfficialOpenSlideReader:
    max_region_nbytes: float = np.inf

    def __init__(self, path: str):
        super().__init__()
        self.slide = openslide.open_slide(path)  # type: ignore

    def get_region(self, box: BoundingBox, scale: float, padding: bool) -> np.ndarray:
        # Find the optimal level and downsample
        target_level = 0
        for level, downsample in enumerate(self.slide.level_downsamples):
            if 1.0 / scale >= downsample:
                target_level = level
        target_downsample = self.slide.level_downsamples[target_level]

        # Read the image
        fetch_size = (
            int(np.ceil(box.width / target_downsample)),
            int(np.ceil(box.height / target_downsample)),
        )
        effective_image = self.slide.read_region(
            location=(box.left, box.top),
            level=target_level,
            size=fetch_size,
        )

        # RGBA -> RGB
        background = Image.new('RGB', effective_image.size, (255, 255, 255))
        background.paste(effective_image, mask=effective_image.split()[3])
        effective_image = np.array(background)

        # Resize to the final scale
        target_size = (
            int(np.ceil(box.width * scale)),
            int(np.ceil(box.height * scale)),
        )
        effective_image = cv2.resize(
            effective_image,
            target_size,
            interpolation=cv2.INTER_LINEAR,
        )

        return effective_image

    @property
    def width(self) -> int:
        return self.slide.dimensions[0]

    @property
    def height(self) -> int:
        return self.slide.dimensions[1]

    @property
    def metadata(self) -> SlideMetadata:
        pixel_width_um = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X, None)
        pixel_height_um = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, None)
        lens = self.slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, None)
        return SlideMetadata(
            width=self.width,
            height=self.height,
            pixel_width_nm=(None if pixel_width_um is None else float(pixel_width_um) * 1000.0),
            pixel_height_nm=(None if pixel_height_um is None else float(pixel_height_um) * 1000.0),
            lens=lens,
        )


def open_slide(path: str) -> OfficialOpenSlideReader:
    return OfficialOpenSlideReader(path)
