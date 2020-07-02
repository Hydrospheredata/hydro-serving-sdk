from collections import namedtuple

from hydrosdk.data.conversions import isinstance_namedtuple


def test_isinstance_namedtuple_namedtuple():
    Point = namedtuple('Point', ['x', 'y'])
    pt = Point(1.0, 5.0)

    assert isinstance_namedtuple(pt)


def test_isinstance_namedtuple_tuple():
    pt = (1, 2, 3)

    assert not isinstance_namedtuple(pt)
