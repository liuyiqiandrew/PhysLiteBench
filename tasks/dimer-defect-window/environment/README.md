# Odd-parity imaging of a dimer sheet

A patterned monolayer has adsorption sites at integer coordinates (x,y), with 0 ≤ x < w and 0 ≤ y < h. Its three undirected bond directions are H=(1,0), V=(0,1) and D=(1,1). Edges end at the physical boundary. The listed vacancy sites cannot bind. In thermal equilibrium **every remaining site belongs to exactly one dimer**. The probability of a covering is proportional to the product of its bond activities. The respective activities are 1, v and z d, in thermal units. The knobs v and d are known. The same unknown z in [0.6,1.4] applies to all sheets.

A phase plate marks a list of distinct bonds. Each occupied marked bond changes the optical phase by π. The dark output records one if the number of occupied marked bonds is odd and zero otherwise. Report its probability for a single equilibrium preparation. Marked bonds can share endpoints. Immobile vacancies are present throughout equilibration.

Each input contains integer w,h in [2,32], v,d in [0.35,2.4], a list `vacancies` of [x,y] pairs, and a list `probe` of one to six [x,y,direction] triples. A triple names the edge from (x,y) along its positive direction. All marked bonds have two available endpoints. The available graph is connected, contains a perfect covering and has at most 32 vacancies. Both even and odd side lengths can occur.

Calibration uses the four corners (0,0), (1,0), (1,1), (2,1) of a rhombus on a 3 by 2 sheet. The plate marks the diagonal from (0,0). The remaining two sites are vacant. For an example outside the calibration protocol, marking both parallel diagonals of this rhombus gives dark probability zero at every knob setting.

Each record in `data/calibration.json` gives an input and its probability rounded to eight decimals. Implement `Model.fit(records)` returning self with the fitted z in `self.activity`. Implement `Model.predict(inputs)` returning a finite, real NumPy array of shape (len(inputs),) in input order. Absolute errors must be below 0.00002 on each calibration reading and below 0.003 on each new probability. The starter treats the environment outside a small binding patch as a reservoir and combines marked bonds independently.
