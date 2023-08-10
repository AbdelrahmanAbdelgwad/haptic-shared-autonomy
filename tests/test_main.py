from src.haptic.__main__ import add
from gym.envs.box2d.car_racing import CarRacing


def test_add():
    assert add(2, 3) == 5


def test_env():
    env = CarRacing(
        allow_reverse=False,
        grayscale=1,
        show_info_panel=1,
        discretize_actions="hard",
        num_tracks=2,
        num_lanes=2,
        num_lanes_changes=4,
        max_time_out=0,
        frames_per_state=4,
    )
    assert 1 == 1
