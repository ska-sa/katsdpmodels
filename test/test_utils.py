################################################################################
# Copyright (c) 2020, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Utilities for other tests."""

import os
import pathlib
import urllib.parse
from typing import ClassVar, Any
from typing_extensions import Literal

import h5py

from katsdpmodels import models


BASE_URL = 'http://test.invalid/data/'
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


def get_data(filename: str) -> bytes:
    with open(os.path.join(DATA_DIR, filename), 'rb') as f:
        return f.read()


def get_data_url(filename: str) -> str:
    return urllib.parse.urljoin(BASE_URL, filename)


def get_file_url(filename: str) -> str:
    path = os.path.join(DATA_DIR, filename)
    return pathlib.PurePath(path).as_uri()


class DummyModel(models.Model):
    model_type: ClassVar[Literal['rfi_mask']] = 'rfi_mask'

    def __init__(self, ranges: Any) -> None:
        self.ranges = ranges
        self.is_closed = False

    @classmethod
    def from_hdf5(cls, hdf5: h5py.File) -> 'DummyModel':
        with hdf5:
            return cls(hdf5['/ranges'][:])

    def close(self) -> None:
        super().close()
        self.is_closed = True
