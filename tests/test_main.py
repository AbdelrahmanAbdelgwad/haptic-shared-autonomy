from haptic.main import add
from gym.envs.box2d.car_racing import CarRacing
from gym.envs.box2d.car_racing import map_val


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


def test_map_val():
    test_case = -1
    assert (
        round(map_val(map_val(test_case, -1, 1, 0, 255), 0, 255, -1, 1), 1) - test_case
    ) <= 0.01
    test_case = -0.8
    assert (
        round(map_val(map_val(test_case, -1, 1, 0, 255), 0, 255, -1, 1), 1) - test_case
    ) <= 0.01
    test_case = -0.6
    assert (
        round(map_val(map_val(test_case, -1, 1, 0, 255), 0, 255, -1, 1), 1) - test_case
    ) <= 0.01
    test_case = -0.4
    assert (
        round(map_val(map_val(test_case, -1, 1, 0, 255), 0, 255, -1, 1), 1) - test_case
    ) <= 0.01
    test_case = -0.2
    assert (
        round(map_val(map_val(test_case, -1, 1, 0, 255), 0, 255, -1, 1), 1) - test_case
    ) <= 0.01
    test_case = 0
    assert (
        round(map_val(map_val(test_case, -1, 1, 0, 255), 0, 255, -1, 1), 1) - test_case
    ) <= 0.01
    test_case = 0.2
    assert (
        round(map_val(map_val(test_case, -1, 1, 0, 255), 0, 255, -1, 1), 1) - test_case
    ) <= 0.01
    test_case = 0.4
    assert (
        round(map_val(map_val(test_case, -1, 1, 0, 255), 0, 255, -1, 1), 1) - test_case
    ) <= 0.01
    test_case = 0.6
    assert (
        round(map_val(map_val(test_case, -1, 1, 0, 255), 0, 255, -1, 1), 1) - test_case
    ) <= 0.01
    test_case = 0.8
    assert (
        round(map_val(map_val(test_case, -1, 1, 0, 255), 0, 255, -1, 1), 1) - test_case
    ) <= 0.01
    test_case = 1
    assert (
        round(map_val(map_val(test_case, -1, 1, 0, 255), 0, 255, -1, 1), 1) - test_case
    ) <= 0.01


if __name__ == "__main__":
    test_cases = [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    for test_case in test_cases:
        print(round(map_val(map_val(test_case, -1, 1, 0, 255), 0, 255, -1, 1), 1))
