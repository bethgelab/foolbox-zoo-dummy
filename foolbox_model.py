import numpy as np


class MockModel:

    def channel_axis(self):
        return 3

    def predictions(self, image):
        lower_bound = self.bounds()[0]
        upper_bound = self.bounds()[1]
        prediction = np.random.randint(lower_bound, upper_bound)
        return prediction

    def bounds(self):
        return (0, 255)

    def num_classes(self):
        return 10


def create():
    net = MockModel()
    return net
