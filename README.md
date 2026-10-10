# conjunction-risk-toolkit
Python toolkit for conjunction risk assessment and collision avoidance.

Assumptions: 

- States and covariances are supplied at TCA.
- Position covariances are 3×3 and expressed in RTN.
- Primary and secondary estimates are assumed independent.
- Covariances are transformed into a common inertial frame before combination.
- Uncertainty is modeled as Gaussian.
- Objects are represented using a combined spherical HBR.
