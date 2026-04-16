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

import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.sophus import SE3  # @manual

from .constants import POSE_DATA_CSV_COLUMNS
from .loader_poses_utils import check_csv_columns
from .pose_utils import lookup_timestamp


@dataclass
class HeadsetPose3d:
    """
    Class to store pose of a headset
    """

    T_world_device: Optional[SE3] = None


HeadsetPose3dTrajectory = Dict[int, HeadsetPose3d]


@dataclass
class HeadsetPose3dWithDt:
    pose3d: HeadsetPose3d
    time_delta_ns: int


class HeadsetPose3dProvider(object):
    def __init__(
        self, headset_pose3d_trajectory: HeadsetPose3dTrajectory, headset_uid: str
    ):
        self._pose3d_trajectory: HeadsetPose3dTrajectory = headset_pose3d_trajectory
        self._sorted_timestamp_ns_list: List[int] = sorted(
            self._pose3d_trajectory.keys()
        )
        self._headset_uid: str = str(headset_uid)

    @property
    def timestamp_ns_list(self) -> List[int]:
        return self._sorted_timestamp_ns_list

    @property
    def headset_uid(self) -> str:
        return self._headset_uid

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Returns the stats of the trajectory
        """
        stats = {}
        stats["num_frames"] = len(self._sorted_timestamp_ns_list)
        stats["headset_uid"] = str(self._headset_uid)
        return stats

    def get_pose_at_timestamp(
        self,
        timestamp_ns: int,
        time_query_options: TimeQueryOptions,
        time_domain: TimeDomain,
        acceptable_time_delta: Optional[int] = None,
    ) -> Optional[HeadsetPose3dWithDt]:
        """
        Return the list of poses at the given timestamp
        """
        if time_domain is not TimeDomain.TIME_CODE:
            raise ValueError("Value other than TimeDomain.TIME_CODE not yet supported.")

        headset_pose3d, time_delta_ns = lookup_timestamp(
            time_indexed_dict=self._pose3d_trajectory,
            sorted_timestamp_list=self._sorted_timestamp_ns_list,
            query_timestamp=timestamp_ns,
            time_query_options=time_query_options,
        )

        if (
            headset_pose3d is None
            or time_delta_ns is None
            or (
                acceptable_time_delta is not None
                and abs(time_delta_ns) > acceptable_time_delta
            )
        ):
            return None
        else:
            return HeadsetPose3dWithDt(
                pose3d=headset_pose3d, time_delta_ns=time_delta_ns
            )


def load_headset_pose_trajectory_from_csv(filename: str) -> Tuple[Dict[int, SE3], str]:
    """Load Device Poses meta data from a CSV file.

    Keyword arguments:
    filename -- the csv file i.e. sequence_folder + "/headset_trajectory.csv"
    """

    pose3d_trajectory: HeadsetPose3dTrajectory = {}
    headset_uids = set()

    # Open the CSV file for reading
    with open(filename, "r") as f:
        reader = csv.reader(f)

        # Read the header row
        header = next(reader)

        # Ensure we have the desired columns
        check_csv_columns(header, POSE_DATA_CSV_COLUMNS)

        # Read the rest of the rows in the CSV file
        for row in reader:
            translation = [
                row[header.index("t_wo_x[m]")],
                row[header.index("t_wo_y[m]")],
                row[header.index("t_wo_z[m]")],
            ]
            quaternion_xyz = [
                row[header.index("q_wo_x")],
                row[header.index("q_wo_y")],
                row[header.index("q_wo_z")],
            ]
            quaternion_w = row[header.index("q_wo_w")]
            timestamp_ns = int(row[header.index("timestamp[ns]")])
            headset_uids.add(str(row[header.index("object_uid")]))

            object_pose = SE3.from_quat_and_translation(
                float(quaternion_w),
                np.array([float(o) for o in quaternion_xyz]),
                np.array([float(o) for o in translation]),
            )[0]

            pose3d_trajectory[timestamp_ns] = HeadsetPose3d(T_world_device=object_pose)

    if len(headset_uids) != 1:
        raise ValueError(
            f"Expected 1 headset pose per timestamp, got {len(headset_uids)}. headset_uids: {headset_uids}"
        )

    return pose3d_trajectory, headset_uids.pop()


def load_headset_pose_provider_from_csv(filename: str) -> HeadsetPose3dProvider:
    """
    Load pose_provider from csv
    """

    headset_pose3d_trajectory, headset_uid = load_headset_pose_trajectory_from_csv(
        filename
    )
    return HeadsetPose3dProvider(
        headset_pose3d_trajectory=headset_pose3d_trajectory,
        headset_uid=headset_uid,
    )
