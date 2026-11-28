from model_api import Detector


class Model(Detector):
    def loop(self, q):
        # Approximate each marked square by an isolated four-bond resonance.
        if q['count'] > 0:
            return 0.
        if q['allow_separate']:
            return 1.
        vertical = (self.activity*q['ratio'])**2
        resonance = 2*vertical/(1+vertical)**2
        return (1-resonance)**2
