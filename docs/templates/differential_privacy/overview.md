# Differential Privacy

This package contains all the elements related to differential privacy. The framework implements the different elements 
allowing use differential privacy isolated or combined with federated learning.

- [Mechanisms](../mechanisms/) contains the main algorithms to apply dp.
- [Sensitivity Sampler](../sensitivity_sampler/) provides a method to estimate epsilon for the 
cases where the privacy algorithm analysis is extremely hard.
- [Composition](../composition/) methods provide some useful methods to deal with composition.
- [Sampling](../sampling/) methods add the option of reduce the amount of privacy consumed with a 
query.
