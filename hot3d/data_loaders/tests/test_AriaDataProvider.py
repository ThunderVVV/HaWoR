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

import os
import unittest

from data_loaders.AriaDataProvider import AriaDataProvider
from projectaria_tools.core.calibration import FISHEYE624, LINEAR
from projectaria_tools.core.stream_id import StreamId

aria_vrs_resource = "aria_sample_recording/sequence.vrs"

try:
    from libfb.py import parutil

    vrs_file_filepath = parutil.get_file_path(
        "test_data/" + aria_vrs_resource, pkg=__package__
    )
except ImportError:
    from pathlib import Path

    THIS_DIR = Path(__file__)
    vrs_file_filepath = str(THIS_DIR.parent / aria_vrs_resource)


class TestAriaDataProvider(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_provider_aria_recording(self) -> None:
        self.assertTrue(os.path.exists(vrs_file_filepath))
        provider = AriaDataProvider(vrs_file_filepath, "")

        self.assertIsNotNone(provider)
        self.assertIsNotNone(provider.get_device_calibration())

        # Aria VRS files contains 3 Image streams (1 RGB, 2 SLAM)
        stream_ids = provider.get_image_stream_ids()
        self.assertEqual(len(stream_ids), 3)
        self.assertTrue(StreamId("214-1") in stream_ids)
        self.assertTrue(StreamId("1201-1") in stream_ids)
        self.assertTrue(StreamId("1201-2") in stream_ids)

        timestamps = provider.get_sequence_timestamps()
        self.assertTrue(len(timestamps) > 0)

        for stream_id in stream_ids:
            img_array = provider.get_image(timestamps[0], stream_id)
            self.assertIsNotNone(img_array)
            if "rgb" in provider.get_image_stream_label(stream_id):
                self.assertEqual(img_array.shape, (1408, 1408, 3))
            elif "slam" in provider.get_image_stream_label(stream_id):
                self.assertEqual(img_array.shape, (480, 640))

            undistorted_img_array = provider.get_image(timestamps[0], stream_id)
            self.assertIsNotNone(undistorted_img_array)

            # Retrieve camera calibration
            self.assertIsNotNone(provider.get_camera_calibration(stream_id))
            self.assertIsNotNone(provider.get_camera_calibration(stream_id, FISHEYE624))
            self.assertIsNotNone(provider.get_camera_calibration(stream_id, LINEAR))

            # Assert we have the right camera type
            self.assertEqual(
                provider.get_camera_calibration(stream_id)[1].model_name(),
                FISHEYE624,
            )
            self.assertEqual(
                provider.get_camera_calibration(stream_id, LINEAR)[1].model_name(),
                LINEAR,
            )

        # MPS resources are empty since not initialized
        self.assertIsNone(provider.get_point_cloud())
