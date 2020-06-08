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

"""Tests for :mod:`katsdpmodels.models`."""

import hashlib

import pytest

from katsdpmodels.models import ensure_str
from test_utils import DummyModel


def test_eq_hash() -> None:
    # model1 and model2 have the same checksum, model3 a different checksum
    model1 = DummyModel(None)
    model2 = DummyModel(None)
    model3 = DummyModel(None)
    model4 = DummyModel(None)
    model1.checksum = hashlib.sha256(b'foo').hexdigest()
    model2.checksum = model1.checksum
    model3.checksum = hashlib.sha256(b'bar').hexdigest()
    assert model1 == model1
    assert model1 == model2
    assert model1 != model3
    assert model3 == model3
    assert model3 != model4
    assert model4 == model4
    assert hash(model1) == hash(model2)
    assert hash(model1) != hash(model3)
    assert hash(model3) != hash(model4)
    assert model1 != 1
    assert model4 != 1


@pytest.mark.parametrize('s', ['foo', b'foo'])
def test_ensure_str(s):
    assert ensure_str(s) == 'foo'


def test_ensure_str_type_error():
    with pytest.raises(TypeError):
        ensure_str(1)
