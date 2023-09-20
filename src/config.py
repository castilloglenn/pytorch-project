from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    c = ConfigDict()

    c.test = "Testing"

    return c
