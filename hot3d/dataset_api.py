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
from typing import Any, Dict, Optional

import data_loaders.HandBox2dDataProvider as HandBox2dDataProvider

import data_loaders.ObjectBox2dDataProvider as ObjectBox2dDataProvider

from data_loaders.AriaDataProvider import AriaDataProvider

from data_loaders.HeadsetPose3dProvider import (
    HeadsetPose3dProvider,
    load_headset_pose_provider_from_csv,
)

from data_loaders.headsets import Headset
from data_loaders.io_utils import load_json
from data_loaders.loader_object_library import ObjectLibrary
from data_loaders.mano_layer import MANOHandModel
from data_loaders.ManoHandDataProvider import MANOHandDataProvider
from data_loaders.HamerManoHandDataProvider import HamerMANOHandDataProvider

from data_loaders.ObjectPose3dProvider import (
    load_pose_provider_from_csv,
    ObjectPose3dProvider,
)
from data_loaders.PathProvider import Hot3dDataPathProvider

from data_loaders.QuestDataProvider import QuestDataProvider

# 3D assets
# - object_uid

# 3D transform
# Aria Device to Optitrack

# Generic idea around the DataProvider is that is allow to initialize the data reading
# and offer a generic interface to retrieve timestamp data by TYPE (Image, Object, Hand, etc.)


class Hot3dDataProvider:
    """
    High Level interface to retrieve and use data from the hot3d dataset
    """

    def __init__(
        self,
        sequence_folder: str,
        object_library: ObjectLibrary,
        mano_hand_model: Optional[MANOHandModel] = None,
        fail_on_missing_data: bool = True,
        use_slam_hamer: bool = False,
    ) -> None:
        """
        INIT_DOC_STRING
        """
        # Will read all required metadata
        # Hands
        # Objects
        # Device type, ...
        self.path_provider = Hot3dDataPathProvider.fromRecordingFolder(
            recording_instance_folderpath=sequence_folder
        )
        if use_slam_hamer:
            self.hamer_mano_hand_pose_trajectory_filepath = self.path_provider.mano_hand_pose_trajectory_filepath.replace("mano_hand_pose_trajectory.jsonl", "slam_hamer.jsonl")
        else:
            self.hamer_mano_hand_pose_trajectory_filepath = None

        if not self.path_provider.is_valid():
            missing_filepaths = [
                filepath
                for filepath in self.path_provider.required_filepaths
                if not os.path.exists(filepath)
            ]

            if fail_on_missing_data:
                raise RuntimeError(
                    f"Invalid hot3d path. Not all expected data are present. missing_filepaths: {missing_filepaths}"
                )
            else:
                print(
                    f"Not all expected data are present. missing_filepaths: {missing_filepaths}"
                )

        self._dynamic_objects_provider = load_pose_provider_from_csv(
            self.path_provider.dynamic_objects_filepath
        )

        self._device_pose_provider = load_headset_pose_provider_from_csv(
            self.path_provider.headset_trajectory_filepath
        )

        self._object_box2d_provider = (
            ObjectBox2dDataProvider.load_box2d_trajectory_from_csv(
                self.path_provider.box2d_objects_filepath
            )
        )

        self._hand_box2d_provider = (
            HandBox2dDataProvider.load_box2d_trajectory_from_csv(
                self.path_provider.box2d_hands_filepath
            )
        )

        self._object_library: ObjectLibrary = object_library

        self._mano_hand_data_provider = None
        if os.path.isfile(self.path_provider.mano_hand_pose_trajectory_filepath):
            if mano_hand_model is not None:
                if self.hamer_mano_hand_pose_trajectory_filepath:
                    self._mano_hand_data_provider = HamerMANOHandDataProvider(
                        self.hamer_mano_hand_pose_trajectory_filepath,
                        mano_hand_model,
                    )
                else:
                    self._mano_hand_data_provider = MANOHandDataProvider(
                        self.path_provider.mano_hand_pose_trajectory_filepath,
                        mano_hand_model,
                    )
            else:
                print("No MANO hand model provided, skipping MANO hand data provider")
        else:
            print(
                f"WARN: Cannot find {self.path_provider.mano_hand_pose_trajectory_filepath}"
            )

        if self.get_device_type() == Headset.Aria:
            self._device_data_provider = AriaDataProvider(
                self.path_provider.vrs_filepath,
                self.path_provider.mps_folderpath,
            )
        elif self.get_device_type() == Headset.Quest3:
            self._device_data_provider = QuestDataProvider(
                self.path_provider.vrs_filepath,
                self.path_provider.camera_models_filepath,
            )
        else:
            raise RuntimeError(f"Unsupported device type {self.get_device_type()}")

    def get_data_statistics(self) -> Dict[str, Any]:
        statistics_dict = {}
        statistics_dict["dynamic_objects"] = (
            self._dynamic_objects_provider.get_data_statistics()
        )

        if self._mano_hand_data_provider is not None:
            statistics_dict["mano_hand_poses"] = (
                self._mano_hand_data_provider.get_data_statistics()
            )

        if self._object_box2d_provider is not None:
            statistics_dict["object_box2ds"] = (
                self.object_box2d_data_provider.get_data_statistics()
            )

        if self._hand_box2d_provider is not None:
            statistics_dict["hand_box2ds"] = (
                self.hand_box2d_data_provider.get_data_statistics()
            )
        return statistics_dict

    @property
    def object_library(self) -> ObjectLibrary:
        """
        Return the object library used for initializing the Hot3dDataProvider
        """
        return self._object_library

    @property
    def device_data_provider(self):
        """
        Return the device data provider (calibration and image stream data)
        """
        return self._device_data_provider

    @property
    def mano_hand_data_provider(self) -> Optional[MANOHandDataProvider]:
        """
        Return the Mano hand data provider
        """
        return self._mano_hand_data_provider

    @property
    def object_box2d_data_provider(self):
        """
        Return the object box2d data provider
        """
        return self._object_box2d_provider

    @property
    def hand_box2d_data_provider(self):
        """
        Return the hand box2d data provider
        """
        return self._hand_box2d_provider

    @property
    def object_pose_data_provider(self) -> Optional[ObjectPose3dProvider]:
        """
        Return the object pose provider
        """
        return self._dynamic_objects_provider

    @property
    def device_pose_data_provider(self) -> Optional[HeadsetPose3dProvider]:
        """
        Return the device pose provider
        """
        return self._device_pose_provider

    def get_device_type(self) -> Headset:
        """
        Return the type of device used for recording (e.g. Quest3, Aria, etc.)
        """
        return Headset[self.get_sequence_metadata()["headset"]]

    def get_sequence_metadata(self) -> Dict:
        """
        Return the metadata associated with the sequence
        """
        metadata_json = load_json(self.path_provider.scene_metadata_filepath)

        return metadata_json
