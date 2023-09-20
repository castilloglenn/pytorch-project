from absl import flags

FLAGS = flags.FLAGS


class Main:
    def __init__(self) -> None:
        print(FLAGS.model.test)
