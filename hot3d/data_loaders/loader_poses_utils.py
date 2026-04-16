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


from typing import List


def check_csv_columns(csv_columns: List[str], expected_columns: List[str]) -> None:
    """Ensure csv_columns is containing all value from expected_columns"""
    for column in csv_columns:
        if column not in expected_columns:
            raise ValueError(
                "Invalid Object CSV format. Expected columns are: {}".format(
                    ", ".join(expected_columns)
                )
            )
