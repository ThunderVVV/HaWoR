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

from typing import List, Optional

import numpy as np
from data_loaders.io_utils import load_json
from PIL import Image

from projectaria_tools.core.calibration import (  # @manual
    CameraCalibration,
    DeviceCadExtrinsics,
    DeviceCalibration,
    distort_by_calibration,
    FISHEYE624,
    get_linear_camera_calibration,
    LINEAR,
)
from projectaria_tools.core.sophus import SE3  # @manual
from projectaria_tools.core.stream_id import StreamId  # @manual

try:
    from pyvrs import ImageConversion, SyncVRSReader  # @manual
except ImportError:
    from pyvrs2 import SyncVRSReader  # @manual
    from vrsbindings import ImageConversion  # @manual


class QuestDataProvider:

    def __init__(self, vrs_filepath: str, device_calibration_filepath: str) -> None:
        self._vrs_reader = SyncVRSReader(vrs_filepath)
        # Configure Image conversion
        self._vrs_reader.set_image_conversion(ImageConversion.NORMALIZE)
        self._vrs_reader.set_stream_type_image_conversion(
            8010, ImageConversion.NORMALIZE_GREY8
        )

        # Keep the two last stream ids (the one with largest resolution)
        image_stream_ids = []
        for stream_id in self._vrs_reader.stream_ids:
            if self._vrs_reader.might_contain_images(stream_id):
                image_stream_ids.append(stream_id)
        image_stream_ids = sorted(image_stream_ids)

        # Filter the reader
        filtered_reader = self._vrs_reader.filtered_by_fields(
            stream_ids=image_stream_ids
        )
        self._vrs_reader = filtered_reader

        # Loading camera calibration data
        device_calibration_json = load_json(device_calibration_filepath)
        camera_calibration = {}
        for it in device_calibration_json:
            quaternion = it["T_Device_Camera"]["quaternion_wxyz"]
            translation = it["T_Device_Camera"]["translation_xyz"]
            image_height = it["imageHeight"]
            image_width = it["imageWidth"]
            label = it["label"]
            max_solid_angle = 1  # Limiting the fov to a constant value.
            # projection_model_type = it["projectionModelType"]
            projection_params = it["projectionParams"]
            serial_number = it["serialNumber"]

            T_world_device = SE3.from_quat_and_translation(
                quaternion[0],
                quaternion[1:4],
                translation,
            )
            # Skip focal_y and rely on a single focal length for x,y
            projection_params = projection_params[:1] + projection_params[2:]

            # Build the corresponding camera calibration object
            camera_calibration[label] = CameraCalibration(
                label,
                FISHEYE624,
                projection_params,
                T_world_device,
                image_width,
                image_height,
                None,
                max_solid_angle,
                serial_number,
            )

        self._device_calibration = DeviceCalibration(
            camera_calibration, {}, {}, {}, {}, DeviceCadExtrinsics(), "", ""
        )

    def get_device_calibration(self) -> DeviceCalibration:
        """
        Return the device calibration (factory calibration of all sensors)
        """
        return self._device_calibration

    def get_camera_calibration(
        self,
        stream_id: StreamId,
        camera_model=FISHEYE624,
    ) -> tuple[SE3, CameraCalibration]:
        """
        Return the camera calibration of the device of the sequence as [Extrinsics, Intrinsics]
        Note:
         - A corresponding pinhole camera can be requested by using camera_model = LINEAR.
         - This is the camera model used to generate the 'get_undistorted_image'.
        """
        if not (camera_model is FISHEYE624 or camera_model is LINEAR):
            raise ValueError(
                "Invalid camera_model type, only FISHEYE624 and LINEAR are supported"
            )

        device_calibration = self.get_device_calibration()
        # Map the string to the right label
        stream_label = self.get_image_stream_label(stream_id)
        stream_labels_str = [
            self.get_image_stream_label(x) for x in self.get_image_stream_ids()
        ]
        idx_stream = stream_labels_str.index(stream_label)
        corresponding_calibration_label = device_calibration.get_camera_labels()[
            idx_stream
        ]
        camera_calibration = device_calibration.get_camera_calib(
            corresponding_calibration_label
        )

        # If a corresponding pinhole camera is requested, we build one on the fly
        if camera_model == LINEAR:
            focal_lengths = camera_calibration.get_focal_lengths()
            image_size = camera_calibration.get_image_size()
            camera_calibration = get_linear_camera_calibration(
                image_size[0], image_size[1], focal_lengths[0]
            )
        # else return the native FISHEYE624 camera model

        T_device_camera = camera_calibration.get_transform_device_camera()
        return [T_device_camera, camera_calibration]

    def get_image_stream_ids(self) -> List[StreamId]:
        # retrieve all streams ids and filter the one that are image based
        image_stream_ids = []
        for stream_id in self._vrs_reader.stream_ids:
            if self._vrs_reader.might_contain_images(stream_id):
                image_stream_ids.append(stream_id)
        image_stream_ids = sorted(image_stream_ids)
        return [StreamId(x) for x in image_stream_ids]

    def get_sequence_timestamps(self) -> List[int]:
        """
        Returns the list of "time code" timestamp for the sequence
        """
        timestamps = self._vrs_reader.get_timestamp_list()
        # convert timestamp from float to int in ns
        return sorted({int(x * 1e9) for x in timestamps})

    def get_image_stream_label(self, stream_id: StreamId) -> str:
        return str(stream_id)

    def get_image(self, timestamp_ns: int, stream_id: StreamId) -> Optional[np.ndarray]:
        try:
            record = self._vrs_reader.read_record_by_time(
                stream_id=self.get_image_stream_label(stream_id),
                timestamp=timestamp_ns / 1e9,
            )
        except ValueError as e:
            print(
                f"No record found for timestamp {timestamp_ns} and stream {stream_id}. Caught exception: {e}"
            )
            record = None

        if record is not None and record.record_type == "data":
            grey8 = Image.fromarray(record.image_blocks[0])
            return np.array(grey8)
        else:
            print(f"No image found for timestamp {timestamp_ns} and stream {stream_id}")
        return None

    def get_undistorted_image(
        self, timestamp_ns: int, stream_id: StreamId
    ) -> Optional[np.ndarray]:
        image = self.get_image(timestamp_ns, stream_id)
        if image is None:
            return None

        [T_device_camera, native_camera_calibration] = self.get_camera_calibration(
            stream_id, camera_model=FISHEYE624
        )
        [T_device_camera, pinhole_camera_calibration] = self.get_camera_calibration(
            stream_id, camera_model=LINEAR
        )

        # Compute the actual undistorted image
        undistorted_image = distort_by_calibration(
            image, pinhole_camera_calibration, native_camera_calibration
        )
        return undistorted_image
