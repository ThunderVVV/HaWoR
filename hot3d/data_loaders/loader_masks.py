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
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from projectaria_tools.core.stream_id import StreamId  # @manual

from .constants import MASK_DATA_CSV_COLUMNS
from .loader_poses_utils import check_csv_columns

TimestampedMask = Dict[int, bool]
StreamMask = Dict[str, TimestampedMask]


class MaskData(object):
    def __init__(self, mask_data: Optional[StreamMask] = None):
        self._mask = mask_data if mask_data is not None else Dict[str, bool]

    @property
    def data(self):
        return self._mask

    @property
    def stream_ids(self):
        return [StreamId(x) for x in self._mask.keys()]

    def stream_mask(self, stream_id: StreamId) -> Optional[TimestampedMask]:
        return self._mask.get(str(stream_id), None)

    def length(self, stream_id: StreamId) -> int:
        if str(stream_id) not in self._mask:
            return 0
        return len(self._mask[str(stream_id)])

    def num_true(self, stream_id: StreamId) -> int:
        """Return the number of True values"""
        if str(stream_id) not in self._mask:
            return 0

        return Counter(self._mask[str(stream_id)].values()).get(True, 0)

    def num_false(self, stream_id: StreamId) -> int:
        """Return the number of False values"""
        if str(stream_id) not in self._mask:
            return 0
        return Counter(self._mask[str(stream_id)].values()).get(False, 0)

    def stats(self):
        return {
            sid: {
                "length": self.length(sid),
                "num_true": self.num_true(sid),
                "num_false": self.num_false(sid),
            }
            for sid in sorted(self._mask.keys())
        }


def load_mask_data(mask_filename: str) -> MaskData:
    """Load mask data from a HOT3D mask CSV file.
    Data saved as CSV with three columns:
    # timestamp[ns],stream_id,mask
    # 67842008213302,214-1,True
    # ...
    """
    mask = {}
    with open(mask_filename, "r") as f:
        reader = csv.reader(f)

        # Read the header row
        header = next(reader)

        # Ensure we have the desired columns
        check_csv_columns(header, MASK_DATA_CSV_COLUMNS)

        # Read the rest of the rows in the CSV file
        for row in reader:
            timestamp_int = row[header.index("timestamp[ns]")]
            stream_id_str = row[header.index("stream_id")]
            value = row[header.index("mask")]

            if stream_id_str not in mask:
                mask[stream_id_str] = {}

            mask[stream_id_str][timestamp_int] = bool(value == "True")

    return MaskData(mask)


def combine_mask_data(
    mask_list: List[MaskData],
    operator: str = "and",  # i.e 'and' or 'or'
) -> MaskData:
    """
    Combine mask data from two or three sources given a logical operator.
    """

    stream_id_strs = {str(y) for x in mask_list for y in x.stream_ids}
    stream_ids = [StreamId(x) for x in stream_id_strs]

    out_mask_dict = {}
    for stream_id in stream_ids:
        timestamped_mask_list = [x.stream_mask(stream_id=stream_id) for x in mask_list]
        if any(x is None for x in timestamped_mask_list):
            raise ValueError("mask data must be present for all streams")
        out_mask_dict[str(stream_id)] = combine_timestamped_mask_data(
            mask_list=timestamped_mask_list, operator=operator
        )
    return MaskData(out_mask_dict)


def combine_timestamped_mask_data(
    mask_list: List[TimestampedMask],
    operator: str = "and",  # i.e 'and' or 'or'
) -> TimestampedMask:

    if len(mask_list) > 0:
        if not all(len(d) == len(mask_list[0]) for d in mask_list):
            raise ValueError("Mask data must have the same length")
    else:
        raise ValueError("mask_list must not be empty")

    ## ensure the timestamps are identical across lists
    reference_tsns_list = list(mask_list[0].keys())
    for it in mask_list[1:]:
        if list(it.keys()) != reference_tsns_list:
            raise ValueError("Mask data must have the same timestamps")

    resulting_array = np.array([mask_list[0][tsns] for tsns in reference_tsns_list])
    # resulting_array = np.array(list(mask_list[0].values()))
    for it in mask_list[1:]:
        if operator == "and":
            resulting_array = resulting_array & np.array(
                [it[tsns] for tsns in reference_tsns_list]
            )
        elif operator == "or":
            resulting_array = resulting_array | np.array(
                [it[tsns] for tsns in reference_tsns_list]
            )
        else:
            raise ValueError("Invalid operator")

    return dict(zip(reference_tsns_list, resulting_array.tolist()))
