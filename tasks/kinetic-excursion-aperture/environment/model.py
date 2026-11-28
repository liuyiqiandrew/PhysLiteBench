"""Free Gaussian bridge approximation to the excursion camera."""
from scipy.special import ndtr
from camera import Camera, middle


class Model(Camera):
    def read(self, q):
        if q["mode"] == "clear":
            return 1.0
        s, r = q["times"]

        def accepted(delta):
            m, std, weight = middle(
                q["launch"], q["incoming"], s, r, q["position"], delta, self.diffusivity
            )
            return weight * ndtr((m - q["cut"]) / std), weight

        if q["mode"] == "direction":
            accepted_part, total = accepted(0.0)
            return accepted_part / total
        plus, _ = accepted(q["kick"])
        minus, _ = accepted(-q["kick"])
        return plus / (plus + minus)
