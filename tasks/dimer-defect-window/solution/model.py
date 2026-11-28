from ports import folded
from contours import reading
from model_api import Detector


class Model(Detector):
    def loop(self, q):
        if q['mode'] == 'fold':
            return folded(q, self.activity)
        return reading(q, self.activity)
