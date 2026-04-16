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
from typing import List, Optional

import numpy as np

from projectaria_tools.core import data_provider  # @manual
from projectaria_tools.core.calibration import (  # @manual
    CameraCalibration,
    DeviceCalibration,
    distort_by_calibration,
    FISHEYE624,
    get_linear_camera_calibration,
    LINEAR,
)
from projectaria_tools.core.mps import (  # @manual
    EyeGaze,
    get_eyegaze_point_at_depth,
    MpsDataPathsProvider,
    MpsDataProvider,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.sophus import SE3  # @manual
from projectaria_tools.core.stream_id import StreamId  # @manual


class AriaDataProvider:

    def __init__(
        self, vrs_filepath: str, mps_folder_path: Optional[str] = None
    ) -> None:
        self._vrs_data_provider = data_provider.create_vrs_data_provider(vrs_filepath)

        # MPS data provider
        if mps_folder_path is not None and os.path.exists(mps_folder_path):
            mps_data_paths_provider = MpsDataPathsProvider(mps_folder_path)
            mps_data_paths = mps_data_paths_provider.get_data_paths()
            self._mps_data_provider = MpsDataProvider(mps_data_paths)
            print(mps_data_paths)
        else:
            self._mps_data_provider = None

    def get_image_stream_ids(self) -> List[StreamId]:
        # retrieve all streams ids and filter the one that are image based
        stream_ids = self._vrs_data_provider.get_all_streams()
        image_stream_ids = [
            p
            for p in stream_ids
            if self._vrs_data_provider.get_label_from_stream_id(p).startswith("camera-")
        ]
        return image_stream_ids

    def get_sequence_timestamps(
        self,
        stream_id: StreamId = StreamId("214-1"),
        time_domain: TimeDomain = TimeDomain.TIME_CODE,
    ) -> List[int]:
        """
        Returns the list of "time code" timestamp for the sequence
        """
        return self._vrs_data_provider.get_timestamps_ns(stream_id, time_domain)

    def get_image_stream_label(self, stream_id: StreamId) -> str:
        return self._vrs_data_provider.get_label_from_stream_id(stream_id)

    def get_image(self, timestamp_ns: int, stream_id: StreamId) -> np.ndarray:
        image = self._vrs_data_provider.get_image_data_by_time_ns(
            stream_id,
            timestamp_ns,
            TimeDomain.TIME_CODE,
            TimeQueryOptions.CLOSEST,
        )
        return image[0].to_numpy_array()

    def get_undistorted_image(
        self, timestamp_ns: int, stream_id: StreamId
    ) -> np.ndarray:
        image = self.get_image(timestamp_ns, stream_id)

        [T_device_camera, native_camera_online_calibration] = (
            self.get_online_camera_calibration(
                stream_id, timestamp_ns=timestamp_ns, camera_model=FISHEYE624
            )
        )
        [T_device_camera, pinhole_camera_online_calibration] = (
            self.get_online_camera_calibration(
                stream_id, timestamp_ns=timestamp_ns, camera_model=LINEAR
            )
        )

        # Compute the actual undistorted image
        undistorted_image = distort_by_calibration(
            image, pinhole_camera_online_calibration, native_camera_online_calibration
        )

        return undistorted_image

    def get_device_calibration(self) -> DeviceCalibration:
        """
        Return the device calibration (factory calibration of all sensors)
        """
        return self._vrs_data_provider.get_device_calibration()

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
        stream_label = self._vrs_data_provider.get_label_from_stream_id(stream_id)
        camera_calibration = device_calibration.get_camera_calib(stream_label)

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

    def get_online_camera_calibration(
        self,
        stream_id: StreamId,
        timestamp_ns: Optional[int],
        time_domain: TimeDomain = TimeDomain.TIME_CODE,
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
        if time_domain is not TimeDomain.TIME_CODE:
            raise ValueError(
                f"{time_domain} is not supported. Only TIME_CODE is supported"
            )

        device_timestamp_ns = (
            self._vrs_data_provider.convert_from_timecode_to_device_time_ns(
                timestamp_ns
            )
        )
        online_calibration = self._mps_data_provider.get_online_calibration(
            device_timestamp_ns=device_timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
        )
        camera_calibs = online_calibration.camera_calibs

        stream_label = self._vrs_data_provider.get_label_from_stream_id(stream_id)
        camera_calib = [c for c in camera_calibs if c.get_label() == stream_label]
        if len(camera_calib) == 0:
            raise ValueError(
                f"camera_calib not found for stream_label: {stream_label} stream_id: {stream_id} at timestamp_ns: {timestamp_ns}"
            )
        camera_calibration = camera_calib[0]

        ## Fix the image size to correspond to the image saved in the vrs.
        ## The calibration returned by mps_data_provider has hardcoded image sizes which are incorrect.
        [_, native_camera_calibration] = self.get_camera_calibration(
            stream_id, camera_model=camera_model
        )
        camera_calibration = CameraCalibration(
            camera_calibration.get_label(),
            camera_model,
            camera_calibration.projection_params(),
            camera_calibration.get_transform_device_camera(),
            native_camera_calibration.get_image_size()[0],  ## correct the image size
            native_camera_calibration.get_image_size()[1],
            camera_calibration.get_valid_radius(),
            camera_calibration.get_max_solid_angle(),
            camera_calibration.get_serial_number(),
        )
        # Store the relative transform from device to camera
        T_device_camera = camera_calibration.get_transform_device_camera()

        # If a corresponding pinhole camera is requested, we build one on the fly
        if camera_model == LINEAR:
            focal_lengths = camera_calibration.get_focal_lengths()
            image_size = camera_calibration.get_image_size()
            camera_calibration = get_linear_camera_calibration(
                image_size[0], image_size[1], focal_lengths[0]
            )
            # Info: transform_device_camera is set to ID in this path in the camera_calibration
        # else return the native FISHEYE624 camera model

        return [T_device_camera, camera_calibration]

    def _timestamp_convert(
        self, timestamp: int, time_domain_in: TimeDomain, time_domain_out: TimeDomain
    ) -> int:
        """
        Returns the converted timestamp between two domains (TimeCode <-> Aria DeviceTime)
        """
        if (
            self._vrs_data_provider
            and time_domain_in == TimeDomain.TIME_CODE
            and time_domain_out == TimeDomain.DEVICE_TIME
        ):
            # Map to corresponding timestamp
            device_timestamp_ns = (
                self._vrs_data_provider.convert_from_timecode_to_device_time_ns(
                    timestamp
                )
            )
            return device_timestamp_ns
        return None

    ###
    # Add MPS data specifics
    ###

    def get_point_cloud(self) -> Optional[np.ndarray]:
        """
        Return the point cloud of the scene
        """
        if self._mps_data_provider is None:
            return None
        if self._mps_data_provider.has_semidense_point_cloud():
            point_cloud_data = self._mps_data_provider.get_semidense_point_cloud()
            # Point cloud filtering is left to the user
            return point_cloud_data

        return None

    def _get_gaze_vector_reprojection(
        self,
        eye_gaze: EyeGaze,
        stream_id_label: str,
        device_calibration: DeviceCalibration,
        camera_calibration: CameraCalibration,
    ) -> np.ndarray:
        """
        Helper function to project a eye gaze output onto a given image and its calibration, assuming specified fixed depth
        """
        gaze_center_in_cpf = get_eyegaze_point_at_depth(
            eye_gaze.yaw, eye_gaze.pitch, depth_m=eye_gaze.depth or 1.0
        )
        transform_device_cpf = device_calibration.get_transform_device_cpf()
        transform_device_camera = device_calibration.get_transform_device_sensor(
            stream_id_label, True
        )
        # We use CAD value (this is the coordinate system used by the Eye Gaze model prediction)
        # Using factory calibration (i.e CAD = False) would lead to less accurate EyeGaze reprojection.
        transform_camera_cpf = transform_device_camera.inverse() @ transform_device_cpf
        gaze_center_in_camera = transform_camera_cpf @ gaze_center_in_cpf
        gaze_center_in_pixels = camera_calibration.project(gaze_center_in_camera)
        return gaze_center_in_pixels

    def get_eye_gaze_in_camera(
        self,
        stream_id: StreamId,
        timestamp_ns: int,
        time_domain: TimeDomain = TimeDomain.TIME_CODE,
        camera_model=FISHEYE624,
    ):
        """
        Return the eye_gaze at the given timestamp projected in the given stream for the given time_domain
        """
        if not (camera_model is FISHEYE624 or camera_model is LINEAR):
            raise ValueError(
                "Invalid camera_model type, only FISHEYE624 and LINEAR are supported"
            )

        eye_gaze = self.get_eye_gaze(timestamp_ns, time_domain)
        if eye_gaze:
            [T_device_camera, camera_calibration] = self.get_camera_calibration(
                stream_id, camera_model
            )
            # Compute eye_gaze vector at depth_m and project it in the image
            gaze_projection = self._get_gaze_vector_reprojection(
                eye_gaze,
                self.get_image_stream_label(stream_id),
                self.get_device_calibration(),
                camera_calibration,
            )
            return gaze_projection
        return None

    def get_eye_gaze(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain = TimeDomain.TIME_CODE,
    ) -> Optional[EyeGaze]:
        """
        Return the eye_gaze data at the given timestamp
        """
        # Map to corresponding timestamp
        if time_domain == TimeDomain.TIME_CODE:
            device_timestamp_ns = self._timestamp_convert(
                timestamp_ns, TimeDomain.TIME_CODE, TimeDomain.DEVICE_TIME
            )
        elif time_domain == TimeDomain.DEVICE_TIME:
            device_timestamp_ns = timestamp_ns
        else:
            raise ValueError("Unsupported time domain")

        if device_timestamp_ns:
            if self._mps_data_provider.has_personalized_eyegaze():
                return self._mps_data_provider.get_personalized_eyegaze(
                    device_timestamp_ns
                )
            elif self._mps_data_provider.has_general_eyegaze():
                return self._mps_data_provider.get_general_eyegaze(device_timestamp_ns)
        return None
