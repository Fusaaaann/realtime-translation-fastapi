import pytest

def pytest_addoption(parser):
    print("added option --use-dummy-mic")
    parser.addoption(
        "--use-dummy-mic",
        action="store_true",
        default=False,
        help="Use dummy microphone data instead of the real audio source."
    )