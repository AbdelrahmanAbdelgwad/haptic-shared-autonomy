from src.haptic.__main__ import add

# from src.haptic.envs.car_racing import CarRacingDiscrete


def test_add():
    assert add(2, 3) == 5
