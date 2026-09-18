import importlib


def test_all_packages_are_importable():
    for name in (
        "app",
        "app.config",
        "app.core",
        "app.routers",
        "app.schemas",
        "app.translate",
    ):
        assert importlib.import_module(name) is not None
