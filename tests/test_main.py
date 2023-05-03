from haptic.__main__ import add
from haptic.env.car_racer_env import subtract

# from haptic.env.car_racer_env import HapticCarRacer


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(3, 2) == 1


# def test_HapticCarRacer():
#     env = HapticCarRacer()
#     action_space = env.action_space.shape
#     assert action_space[0] == 3
