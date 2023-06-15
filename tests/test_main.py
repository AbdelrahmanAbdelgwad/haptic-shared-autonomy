from haptic.__main__ import add
from haptic.envs.car_racing import CarRacingDiscrete


def test_add():
    assert add(2, 3) == 5
