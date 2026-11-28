"""Three independent copies of the calibrated avalanche share the same gain and timing selection."""
from current import Calibrated
from single_gate import spectral_law
from transport import packet_law


class Model(Calibrated):
    def waveform(self, q):
        base, moment = packet_law(q, self.activity, 9)
        weight = base.sum()
        if q["observable"] == "waveform":
            return moment.sum() / weight

        def marginal(i, part):
            spec = dict(window=q["gates"][i], charge=q["charges"][i], part=part)
            return (
                spectral_law(dict(q, spectrum=spec), self.activity, 9)[0].sum() / weight
            )

        if q["part"] == "point":
            return marginal(0, "line") * marginal(1, "line")
        if q["part"] == "area":
            return marginal(0, "density") * marginal(1, "density")
        return marginal(0, "line") * marginal(1, "density") + marginal(
            0, "density"
        ) * marginal(1, "line")
