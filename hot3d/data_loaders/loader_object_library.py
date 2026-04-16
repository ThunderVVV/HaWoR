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
from typing import Any, Dict, Set

from .io_utils import load_json


class ObjectLibrary(object):
    def __init__(self, object_library_json: Dict[str, Any], asset_folder: str):
        self._object_library_json = object_library_json
        (
            self._object_id_to_name_dict,
            self._object_name_to_id_dict,
        ) = self._get_object_id_name_mappings(object_library_json)
        (
            self._headset_id_to_name_dict,
            self._headset_name_to_id_dict,
        ) = self._get_headset_id_name_mappings(object_library_json)

        self._asset_folder = asset_folder

    @property
    def object_id_to_name_dict(self):
        return self._object_id_to_name_dict

    @property
    def object_name_to_id_dict(self):
        return self._object_name_to_id_dict

    @property
    def headset_id_to_name_dict(self):
        return self._headset_id_to_name_dict

    @property
    def headset_name_to_id_dict(self):
        return self._headset_name_to_id_dict

    @property
    def object_uids(self) -> Set[str]:
        return set(self._object_name_to_id_dict.values())

    @property
    def headset_uids(self) -> Set[str]:
        return set(self._headset_name_to_id_dict.values())

    @property
    def asset_folder_name(self) -> str:
        return self._asset_folder

    def _get_object_id_name_mappings(self, object_info_json):

        object_id_to_name_dict = {
            k: v["instance_name"]
            for k, v in object_info_json.items()
            if v["instance_type"] == "object" and v["motion_type"] == "dynamic"
        }
        object_name_to_id_dict = {v: k for k, v in object_id_to_name_dict.items()}
        return object_id_to_name_dict, object_name_to_id_dict

    def _get_headset_id_name_mappings(self, object_info_json):
        headset_id_to_name_dict = {
            k: v["instance_name"]
            for k, v in object_info_json.items()
            if v["instance_type"] == "headset"
        }
        headset_name_to_id_dict = {v: k for k, v in headset_id_to_name_dict.items()}
        return headset_id_to_name_dict, headset_name_to_id_dict

    @staticmethod
    def get_cad_asset_path(object_library_folderpath: str, object_id: str) -> str:
        return os.path.join(object_library_folderpath, f"{object_id}.glb")


def load_object_library(object_library_folderpath: str) -> ObjectLibrary:
    instance_filepath = os.path.join(object_library_folderpath, "instance.json")
    object_library_json = load_json(instance_filepath)
    asset_folder = os.path.join(object_library_folderpath)
    return ObjectLibrary(object_library_json, asset_folder)
