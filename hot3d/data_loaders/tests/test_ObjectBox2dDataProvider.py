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

from data_loaders.ObjectBox2dDataProvider import load_box2d_trajectory_from_csv
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.stream_id import StreamId  # @manual

box2d_objects_resource = "aria_sample_recording/box2d_objects.csv"

try:
    from libfb.py import parutil

    box2d_objects_filepath = parutil.get_file_path(
        "test_data/" + box2d_objects_resource, pkg=__package__
    )
except ImportError:
    from pathlib import Path

    THIS_DIR = Path(__file__)
    box2d_objects_filepath = str(THIS_DIR.parent / box2d_objects_resource)


class TestObjectBox2dDataProvider(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_provider_aria_recording(self) -> None:
        self.assertTrue(os.path.exists(box2d_objects_filepath))
        provider = load_box2d_trajectory_from_csv(filename=box2d_objects_filepath)
        object_uids = provider.object_uids
        self.assertEqual(len(object_uids), 6)

        stream_ids = provider.stream_ids
        self.assertEqual(len(stream_ids), 3)

        self.assertTrue(StreamId("214-1") in stream_ids)
        self.assertTrue(StreamId("1201-1") in stream_ids)
        self.assertTrue(StreamId("1201-2") in stream_ids)

        for stream_id in stream_ids:
            timestamp_ns_list = provider.get_timestamp_ns_list(stream_id=stream_id)
            self.assertIsNotNone(timestamp_ns_list)
            self.assertGreater(len(timestamp_ns_list), 0)

            query_timestamp_ns = timestamp_ns_list[len(timestamp_ns_list) // 2]

            box2d_collection_with_dt = provider.get_box2d_at_timestamp(
                stream_id=stream_id,
                timestamp_ns=query_timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )

            self.assertIsNotNone(box2d_collection_with_dt)
            self.assertIsNotNone(box2d_collection_with_dt.box2d_collection)
            object_uids_at_query_timestamp = (
                box2d_collection_with_dt.box2d_collection.object_uid_list
            )
            self.assertGreater(len(object_uids_at_query_timestamp), 0)
