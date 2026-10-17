Infer the common speed from the calibration data and implement the rod-current
model in `/app/model.py`. Use `/app/README.md` for the apparatus and API.
The fitted model must predict all three readouts throughout the stated input
domain. You may change the supplied response model. Keep the public API.

Additional guidance:
The exponential current average includes the freshly drawn rod-length order. A large current can select an atypical composition of the net crossed block. Retain the probability cost of that composition together with the free-particle crossing cost before optimizing; replacing the crossed lengths by their ordinary mean loses this contribution.
