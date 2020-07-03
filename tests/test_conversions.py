from collections import namedtuple

import pandas as pd

from hydrosdk.data.conversions import isinstance_namedtuple


def test_isinstance_namedtuple_namedtuple():
    Point = namedtuple('Point', ['x', 'y'])
    pt = Point(1.0, 5.0)

    assert isinstance_namedtuple(pt)


def test_isinstance_namedtuple_tuple():
    pt = (1, 2, 3)

    assert not isinstance_namedtuple(pt)


def test_isinstance_namedtuple_itertuples():
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)

    for row in df.itertuples():
        assert isinstance_namedtuple(row)
