# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np


class AlignedBox2d:
    """
    An 2D axis aligned box in floating point.

    Assumptions:
    * The origin is at top-left corner
    * `right` and `bottom` are not inclusive in the region, i.e. the width and
        height can be simply calculated by `right - left` and `bottom - top`.
    """

    def __init__(self, left: float, top: float, right: float, bottom: float):
        """Initializes the bounding box given (left, top, right, bottom)"""
        self._left: float = left
        self._top: float = top
        self._right: float = right
        self._bottom: float = bottom

    def __repr__(self):
        return f"AlignedBox2d(left: {self._left}, top: {self._top}, right: {self._right}, bottom: {self._bottom})"

    @property
    def left(self) -> float:
        """Left of the aligned box on x-axis."""
        return self._left

    @property
    def top(self) -> float:
        """Top of the aligned box on y-axis."""
        return self._top

    @property
    def right(self) -> float:
        """Right of the aligned box on x-axis."""
        return self._right

    @property
    def bottom(self) -> float:
        """Bottom of the aligned box on y-axis."""
        return self._bottom

    @property
    def width(self) -> float:
        """Width of the aligned box.

        Returns:
            Width computed by right - left
        """
        return self.right - self.left

    @property
    def height(self) -> float:
        """Height of the aligned box.

        Returns:
            Height computed by bottom - top
        """
        return self.bottom - self.top

    def pad(self, width: float, height: float) -> AlignedBox2d:
        """Pads the region by extending `width` and `height` on four sides.

        Args:
            width (float): length to pad on left and right sides
            height (float): length to pad on top and bottom sides
        Returns:
            a new AlignedBox2d object with padded region
        """
        return AlignedBox2d(
            self.left - width,
            self.top - height,
            self.right + width,
            self.bottom + height,
        )

    def array_ltrb(self) -> np.ndarray:
        """Converts the box into a float np.ndarray of shape (4,):  (left, top, right, bottom).

        Returns:
            a float np.ndarray of shape (4,) representing (left, top, right, bottom)
        """
        return np.array([self.left, self.top, self.right, self.bottom])

    def array_ltwh(self) -> np.ndarray:
        """Converts the box into a float np.ndarray of shape (4,): (left, top, width, height).

        Returns:
            a float np.ndarray of shape (4,) representing (left, top, width, height)
        """
        return np.array([self.left, self.top, self.width, self.height])

    def int_array_ltrb(self) -> np.ndarray:
        """Converts the box into an int np.ndarray of shape (4,): (left, top, width, height).

        Returns:
            an int np.ndarray of shape (4,) representing (left, top, right, bottom)
        """
        return self.array_ltrb().astype(int)

    def int_array_ltwh(self) -> np.ndarray:
        """Converts the box into an int np.ndarray of shape (4,): (left, top, width, height).

        Returns:
            an int np.ndarray of shape (4,) representing (left, top, width, height)
        """
        return self.array_ltwh().astype(int)

    def round(self) -> AlignedBox2d:
        """Rounds the float values to int.

        Returns:
            a new AlignedBox2d object with rounded values (still float)
        """
        return AlignedBox2d(
            np.round(self.left),
            np.round(self.top),
            np.round(self.right),
            np.round(self.bottom),
        )

    def clip(self, boundary: AlignedBox2d) -> AlignedBox2d:
        """Clips the region by the boundary

        Args:
            boundary (AlignedBox2d): boundary of box to be clipped
                (boundary.left: minimum left / right value,
                 boundary.top: minimum top / bottom value,
                 boundary.right: maximum left / right value,
                 boundary.bottom: maximum top / bottom value)
        Returns:
            a new clipped AlignedBox2d object
        """
        return AlignedBox2d(
            min(max(self.left, boundary.left), boundary.right),
            min(max(self.top, boundary.top), boundary.bottom),
            min(max(self.right, boundary.left), boundary.right),
            min(max(self.bottom, boundary.top), boundary.bottom),
        )
