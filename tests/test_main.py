from haptic.env.car_racer_env import HapticCarRacer


def test_HapticCarRacer():
    env = HapticCarRacer()
    action_space = env.action_space.shape
    assert action_space == (3,)
