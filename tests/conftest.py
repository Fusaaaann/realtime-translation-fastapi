import pytest


def pytest_addoption(parser):
    parser.addoption("--use-real-mic", action="store_true", default=False, help="Use real audio source instead of the dummy microphone data.")
    parser.addoption("--use-real-api", action="store_true", default=False, help="Call third party api instead of returning mocked data.")


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
