from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    c = ConfigDict()

    c.batch_size = 64
    c.epochs = 5

    return c
