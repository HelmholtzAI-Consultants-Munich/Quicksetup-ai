import pytest

from tests.helpers.run_command import run_command


@pytest.mark.slow
def test_debug_default():
    command = ["scripts/train.py", "debug=default"]
    run_command(command)


def test_debug_limit_batches():
    command = ["scripts/train.py", "debug=limit_batches"]
    run_command(command)


def test_debug_overfit():
    command = ["scripts/train.py", "debug=overfit"]
    run_command(command)


@pytest.mark.slow
def test_debug_profiler():
    command = ["scripts/train.py", "debug=profiler"]
    run_command(command)


def test_debug_step():
    command = ["scripts/train.py", "debug=step"]
    run_command(command)


def test_debug_test_only():
    command = ["scripts/train.py", "debug=test_only"]
    run_command(command)
