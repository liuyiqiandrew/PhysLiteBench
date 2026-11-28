import numpy as np
from scipy.sparse.csgraph import dijkstra


class Camera:
    def fit(self,rows):
        t=np.array([r["input"]["test_round"] for r in rows])
        self.rate=float(-np.dot(t,np.log([r["value"] for r in rows]))/np.dot(t,t))
        return self

    def measure(self,q):
        # Travel along open tracks from the drive on the full undamaged graph.
        graph=np.zeros((q["sites"],q["sites"]))
        for a,b,l in q["tracks"]:
            if min(a,b)>=0:graph[a,b]=graph[b,a]=l
        d=dijkstra(graph,indices=q["drive"])
        t=[]
        for e,r in q["pickup"]:
            a,b,l=q["tracks"][e]
            t.append(min(d[a]+r*l if a>=0 else np.inf,d[b]+(1-r)*l if b>=0 else np.inf))
        g,p=t
        return float(q["delay"][0]<=p-g<=q["delay"][1])*np.exp(-self.rate*max(0,p-g))

    def predict(self,inputs):
        return np.asarray([np.exp(-self.rate*q["test_round"]) if "test_round" in q
                           else self.measure(q) for q in inputs],dtype=float)
