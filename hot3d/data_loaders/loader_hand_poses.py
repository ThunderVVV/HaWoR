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

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from projectaria_tools.core.sophus import SE3  # @manual

LEFT_HAND_INDEX = 0
RIGHT_HAND_INDEX = 1


class Handedness(Enum):
    Left = int(LEFT_HAND_INDEX)
    Right = int(RIGHT_HAND_INDEX)


@dataclass
class HandPose:
    """Define a Hand pose as wrist_pose (SE3), and joint_angles."""

    handedness: Handedness
    wrist_pose: Optional[SE3]
    joint_angles: List[float]

    def is_left_hand(self) -> bool:
        return self.handedness == Handedness.Left

    def is_right_hand(self) -> bool:
        return self.handedness == Handedness.Right

    def handedness_label(self) -> str:
        return "left" if self.is_left_hand() else "right"


@dataclass
class HandPose3dCollection:
    """
    Class to store the Hand poses for a given timestamp
    """

    timestamp_ns: int
    poses: Dict[Handedness, HandPose]


TimestampHandPoses3d = Dict[int, HandPose3dCollection]


def _get_hand_pose(handedness: str, hand_poses_json: Dict) -> Optional[SE3]:
    if handedness in hand_poses_json.keys():
        wrist = hand_poses_json[handedness]["wrist_xform"]
        quaternion_w = wrist["q_wxyz"][0]
        quaternion_xyz = wrist["q_wxyz"][1:4]
        translation = wrist["t_xyz"]
        hand_pose = SE3.from_quat_and_translation(
            float(quaternion_w),
            np.array([float(o) for o in quaternion_xyz]),
            np.array([float(o) for o in translation]),
        )[0]
        return hand_pose
    return None


def _get_mano_joint_angles(
    handedness: str, hand_poses_json: Dict
) -> Optional[List[float]]:
    if handedness in hand_poses_json.keys():
        pose_pca = hand_poses_json[handedness]["pose"]
        return pose_pca
    return None


def parse_hand_poses_from_fileobject(fileobject):
    hand_poses_per_timestamp: TimestampHandPoses3d = {}
    for line in fileobject:
        # Parse the JSON file line
        hand_pose_instance = json.loads(line)
        timestamp_ns = hand_pose_instance["timestamp_ns"]
        hand_poses_json = hand_pose_instance["hand_poses"]

        # Read hand pose data
        left_hand_pose = _get_hand_pose(str(LEFT_HAND_INDEX), hand_poses_json)
        right_hand_pose = _get_hand_pose(str(RIGHT_HAND_INDEX), hand_poses_json)

        left_joint_angles = _get_mano_joint_angles(
            str(LEFT_HAND_INDEX), hand_poses_json
        )
        right_joint_angles = _get_mano_joint_angles(
            str(RIGHT_HAND_INDEX), hand_poses_json
        )

        # If hand pose data is available, add it to the dictionary
        if (
            left_hand_pose is not None or right_hand_pose is not None
        ) and timestamp_ns not in hand_poses_per_timestamp:
            hand_poses_per_timestamp[timestamp_ns] = HandPose3dCollection(
                timestamp_ns=timestamp_ns, poses={}
            )

        if left_hand_pose is not None:
            hand_poses_per_timestamp[timestamp_ns].poses[Handedness.Left] = HandPose(
                Handedness.Left, left_hand_pose, left_joint_angles
            )

        if right_hand_pose is not None:
            hand_poses_per_timestamp[timestamp_ns].poses[Handedness.Right] = HandPose(
                Handedness.Right, right_hand_pose, right_joint_angles
            )
    return hand_poses_per_timestamp

def parse_hamer_hand_poses_from_fileobject(fileobject):
    hand_poses_per_timestamp: TimestampHandPoses3d = {}
    for line in fileobject:
        # Parse the JSON file line
        hand_pose_instance = json.loads(line)
        timestamp_ns = hand_pose_instance["timestamp_ns"]
        hand_poses_json = hand_pose_instance["hand_poses"]

        # Read hand pose data
        left_hand_pose = _get_hand_pose(str(LEFT_HAND_INDEX), hand_poses_json)
        right_hand_pose = _get_hand_pose(str(RIGHT_HAND_INDEX), hand_poses_json)

        left_joint_angles = _get_mano_joint_angles(
            str(LEFT_HAND_INDEX), hand_poses_json
        )
        right_joint_angles = _get_mano_joint_angles(
            str(RIGHT_HAND_INDEX), hand_poses_json
        )

        # If hand pose data is available, add it to the dictionary
        if (
            left_hand_pose is not None or right_hand_pose is not None
        ) and timestamp_ns not in hand_poses_per_timestamp:
            hand_poses_per_timestamp[timestamp_ns] = HandPose3dCollection(
                timestamp_ns=timestamp_ns, poses={}
            )

        if left_hand_pose is not None:
            hand_poses_per_timestamp[timestamp_ns].poses[Handedness.Left] = HandPose(
                Handedness.Left, left_hand_pose, left_joint_angles
            )

        if right_hand_pose is not None:
            hand_poses_per_timestamp[timestamp_ns].poses[Handedness.Right] = HandPose(
                Handedness.Right, right_hand_pose, right_joint_angles
            )
    return hand_poses_per_timestamp


def load_hand_poses(filename: str) -> TimestampHandPoses3d:
    """Load Hand Poses meta data from a JSONL file.

    Keyword arguments:
    filename -- the jsonl file i.e. sequence_folder + "/hand_pose_trajectory.jsonl"
    """
    with open(filename, "r") as fobj:
        hand_poses_per_timestamp = parse_hand_poses_from_fileobject(fobj)

    return hand_poses_per_timestamp

def load_hamer_hand_poses(filename: str) -> TimestampHandPoses3d:
    """Load Hand Poses meta data from a JSONL file.

    Keyword arguments:
    filename -- the jsonl file i.e. sequence_folder + "/hand_pose_trajectory.jsonl"
    """
    with open(filename, "r") as fobj:
        hand_poses_per_timestamp = parse_hamer_hand_poses_from_fileobject(fobj)

    return hand_poses_per_timestamp


def load_mano_shape_params(filename: str) -> Optional[List[float]]:
    betas = None
    with open(filename, "rb") as f:
        for line in f:
            hand_pose_instance = json.loads(line)
            for handedness in ["0", "1"]:
                if (
                    handedness in hand_pose_instance["hand_poses"].keys()
                    and "betas" in hand_pose_instance["hand_poses"][handedness]
                ):
                    betas = hand_pose_instance["hand_poses"][handedness]["betas"]
                    break
            if betas is not None:
                break
    return betas
