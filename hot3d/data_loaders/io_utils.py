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
from typing import Any, Optional


def load_json(json_filepath):
    with open(json_filepath, "r") as fp:
        return json.loads(fp.read())


def write_json(payload, json_filepath):
    with open(json_filepath, "w") as fp:
        json.dump(payload, fp, indent=4, sort_keys=True)


def is_float(x: Any) -> bool:
    """
    Function checks if the input is convertible to float
    """
    if x is None:
        return False
    if len(x) == 0:
        return False
    try:
        float(x)
        return True
    except ValueError:
        return False


def float_or_none(x: Any) -> Optional[float]:
    """
    Function returns a float if x is convertible to float, otherwise None
    """
    return float(x) if is_float(x) else None
