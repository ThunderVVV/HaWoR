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
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.stream_id import StreamId  # @manual

from .AlignedBox2d import AlignedBox2d

from .constants import BOX2D_DATA_CSV_COLUMNS
from .io_utils import float_or_none, is_float
from .loader_poses_utils import check_csv_columns
from .pose_utils import lookup_timestamp

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s",
)


@dataclass
class ObjectBox2d:
    box2d: Optional[AlignedBox2d]
    visibility_ratio: float


@dataclass
class ObjectBox2dCollection:
    timestamp_ns: int
    box2ds: Dict[str, ObjectBox2d]

    @property
    def object_uid_list(self) -> Set[str]:
        return set(self.box2ds.keys())


ObjectBox2dTrajectory = Dict[
    int, ObjectBox2dCollection
]  # trajectory for a single stream
ObjectBox2dTrajectoryCollection = Dict[
    str, ObjectBox2dTrajectory
]  # trajectories for multiple streams


@dataclass
class ObjectBox2dCollectionWithDt:
    box2d_collection: ObjectBox2dCollection
    time_delta_ns: int


class ObjectBox2dProvider:

    def __init__(
        self, box2d_trajectory_collection: ObjectBox2dTrajectoryCollection
    ) -> None:

        self._box2d_trajectory_collection = box2d_trajectory_collection

        self._sorted_timestamp_ns_list: Dict[str, List[int]] = {}
        for stream_id in self._box2d_trajectory_collection.keys():
            self._sorted_timestamp_ns_list[stream_id] = sorted(
                self._box2d_trajectory_collection[stream_id].keys()
            )

        self._object_uids_with_box2ds: Set = {
            x
            for box2d_trajectory in self._box2d_trajectory_collection.values()
            for v in box2d_trajectory.values()
            for x in v.object_uid_list
        }

    def get_timestamp_ns_list(self, stream_id: StreamId) -> Optional[List[int]]:
        return self._sorted_timestamp_ns_list.get(str(stream_id), None)

    @property
    def stream_ids(self) -> List[StreamId]:
        return [StreamId(x) for x in self._box2d_trajectory_collection.keys()]

    @property
    def object_uids(self) -> Set[str]:
        return set(self._object_uids_with_box2ds)

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Returns the stats of the trajectory
        """
        stats = {}
        stats["num_frames"] = {
            k: sum(v) for k, v in self._sorted_timestamp_ns_list.items()
        }
        stats["stream_ids"] = [str(x) for x in self.stream_ids]
        stats["num_objects"] = len(self.object_uids)
        stats["object_uids"] = [str(x) for x in self.object_uids]
        return stats

    def get_box2d_at_timestamp(
        self,
        stream_id: StreamId,
        timestamp_ns: int,
        time_query_options: TimeQueryOptions,
        time_domain: TimeDomain,
        acceptable_time_delta: Optional[int] = None,
    ) -> Optional[ObjectBox2dCollectionWithDt]:
        """
        Return the list of boxes at the given timestamp
        """
        if time_domain is not TimeDomain.TIME_CODE:
            raise ValueError("Value other than TimeDomain.TIME_CODE not yet supported.")

        if stream_id not in self.stream_ids:
            raise ValueError(f"Box2d trajectory not available for stream {stream_id}.")

        box2d_collection, time_delta_ns = lookup_timestamp(
            time_indexed_dict=self._box2d_trajectory_collection[str(stream_id)],
            sorted_timestamp_list=self.get_timestamp_ns_list(stream_id=stream_id),
            query_timestamp=timestamp_ns,
            time_query_options=time_query_options,
        )

        if (
            box2d_collection is None
            or time_delta_ns is None
            or (
                acceptable_time_delta is not None
                and abs(time_delta_ns) > acceptable_time_delta
            )
        ):
            return None
        else:
            return ObjectBox2dCollectionWithDt(
                box2d_collection=box2d_collection, time_delta_ns=time_delta_ns
            )


def parse_box2ds_from_csv_reader(csv_reader) -> ObjectBox2dTrajectoryCollection:

    box2d_trajectory_collection: ObjectBox2dTrajectoryCollection = {}

    # Read the header row
    header = next(csv_reader)

    # Ensure we have the desired columns
    check_csv_columns(header, BOX2D_DATA_CSV_COLUMNS)

    # Read the rest of the rows in the CSV file
    for row in csv_reader:

        stream_id = str(StreamId(row[header.index("stream_id")]))
        timestamp_ns = int(row[header.index("timestamp[ns]")])
        object_uid = str(row[header.index("object_uid")])
        visibility_ratio = float_or_none(row[header.index("visibility_ratio[%]")])

        if is_float(row[header.index("x_min[pixel]")]):
            x_min_px = float(row[header.index("x_min[pixel]")])
            x_max_px = float(row[header.index("x_max[pixel]")])
            y_min_px = float(row[header.index("y_min[pixel]")])
            y_max_px = float(row[header.index("y_max[pixel]")])

            box2d = AlignedBox2d(
                left=x_min_px, top=y_min_px, right=x_max_px, bottom=y_max_px
            )
        else:
            box2d = None

        object_box2d = ObjectBox2d(
            box2d=box2d,
            visibility_ratio=visibility_ratio,
        )

        if stream_id not in box2d_trajectory_collection:
            box2d_trajectory_collection[stream_id] = {}

        if timestamp_ns not in box2d_trajectory_collection[stream_id]:
            box2d_trajectory_collection[stream_id][timestamp_ns] = (
                ObjectBox2dCollection(timestamp_ns=timestamp_ns, box2ds={})
            )

        box2d_trajectory_collection[stream_id][timestamp_ns].box2ds[
            object_uid
        ] = object_box2d
    return box2d_trajectory_collection


def load_box2d_trajectory_from_csv(filename: str) -> Optional[ObjectBox2dProvider]:
    """Load Dynamic Objects meta data from a CSV file.

    Keyword arguments:
    filename -- the csv file i.e. sequence_folder + "/dynamic_objects.csv"
    """
    if not os.path.exists(filename):
        logger.warn(f"filename: {filename} does not exist.")
        return None

    # Open the CSV file for reading
    with open(filename, "r") as f:
        csv_reader = csv.reader(f)
        box2d_trajectory_collection = parse_box2ds_from_csv_reader(
            csv_reader=csv_reader
        )
        return ObjectBox2dProvider(
            box2d_trajectory_collection=box2d_trajectory_collection
        )
