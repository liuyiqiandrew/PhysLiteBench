from apparatus import Readout
from histories import measurement
from lead import lead_gram


class Model(Readout):
    """Before a regional charge measurement, also resolve every site occupation."""
    def fringe(self,q):
        return measurement(q,lambda lead,paths: lead_gram(lead,paths,record_sites=True))[1]
