"""v4.1: the test cell's two image checks, a barcode scan and a visual inspection
(docs/v4_1_plan.md, section 1; simulation/test_cell.py has the cameras and the modules' looks).

Both cameras look at a module's top from its outer side, 45 deg down, 0.28 m away, through an
18 deg lens (96 x 96 pixels, about 1 mm per pixel at the module). Every image goes through
`ImagingNoise` first: an exposure that varies from image to image (the ring light, the module's
height and tilt in the hand-off, ageing) and sensor noise per pixel. The rules then read only the
image, by colour, inside the central 70 % of it (the module's top is there; the edges show the
bench and whatever stands behind it):

- **label**: the share of white pixels (every channel at least 200, the channels within 40 of
  each other). A module's label covers 2-4 % of the window; none, 0 %. A label is seen if the
  share is at least `label_min` (1 %).
- **red mark**: the share of strongly red pixels (red at least 140 and more than twice green and
  blue). A mark is seen at `red_min`.

The **scan** (at the conveyor's pick point) reads the barcode if the label is seen. It does not
decode the bars: in the simulation a seen label stands for a good read, and the serial read is
the module's own. A module whose label is missing cannot be read: "no read".

The **visual inspection** (at the vision station, after the test) finds a visible defect if the
label is not seen or a red mark is.

**Imperfect, the way such a rule is.** It is a fixed colour threshold on a small, noisy image:
- a small red mark (the drawn severity: down to 2.4 x 1.6 mm, a few pixels) falls under the
  threshold, more often the smaller it is and the darker the exposure: an **escape** if the
  module passes its electrical test;
- harmless brownish flux residue is near the rule's idea of red; in a bright exposure, with
  noise, enough of its pixels pass the ratio test to read as a mark: a **false reject**.
No rule is tuned on shift results. The thresholds and the noise were set from renders of the
drawn looks only (tests/test_test_cell.py checks the separation at full size and that a faint
mark or residue lands in between).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InspectionRule:
    window: float = 0.7           # the central share of the image (each way) that is read
    white_min: int = 200          # every channel at least this
    white_spread: int = 40        # channels within this of each other
    red_min_value: int = 140
    red_ratio: float = 2.0        # red more than this times green and blue
    label_min: float = 0.01       # share of white pixels that is a label
    red_min: float = 0.003        # share of red pixels that is a mark


RULE = InspectionRule()


@dataclass(frozen=True)
class ImagingNoise:
    """What makes two images of the same module differ: a per-image exposure gain (log-normal,
    `gain` its standard deviation) and per-pixel sensor noise (`pixel` 8-bit levels, Gaussian).
    `none()` for a clean image."""
    gain: float = 0.06
    pixel: float = 4.0

    @classmethod
    def none(cls) -> ImagingNoise:
        return cls(0.0, 0.0)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        g = float(np.exp(rng.normal(0.0, self.gain))) if self.gain > 0 else 1.0
        f = np.asarray(image[..., :3], dtype=float) * g
        if self.pixel > 0:
            f = f + rng.normal(0.0, self.pixel, f.shape)
        return np.clip(np.rint(f), 0, 255).astype(np.uint8)


NOISE = ImagingNoise()


@dataclass(frozen=True)
class Inspection:
    white: float                  # share of white pixels in the window
    red: float                    # share of red pixels in the window
    label: bool                   # a label is seen
    mark: bool                    # a red mark is seen

    @property
    def readable(self) -> bool:
        """The barcode scan reads the module."""
        return self.label

    @property
    def defect(self) -> bool:
        """The visual inspection finds a visible defect."""
        return not self.label or self.mark

    def as_dict(self) -> dict:
        return {"white": round(self.white, 4), "red": round(self.red, 4), "label": self.label,
                "mark": self.mark}


def inspect(image: np.ndarray, rule: InspectionRule = RULE) -> Inspection:
    """Read an RGB image (H x W x 3, uint8) with the rule."""
    h, w = image.shape[:2]
    mh, mw = int(h * (1 - rule.window) / 2), int(w * (1 - rule.window) / 2)
    f = np.asarray(image[mh:h - mh, mw:w - mw, :3], dtype=int)
    lo, hi = f.min(-1), f.max(-1)
    white = float(((lo >= rule.white_min) & (hi - lo <= rule.white_spread)).mean())
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    red = float(((r >= rule.red_min_value) & (r > rule.red_ratio * g)
                 & (r > rule.red_ratio * b)).mean())
    return Inspection(white, red, white >= rule.label_min, red >= rule.red_min)
