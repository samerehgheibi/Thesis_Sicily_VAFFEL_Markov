[markov chain1.py](https://github.com/user-attachments/files/23209313/markov.chain1.py)# Thesis_Sicily_VAFFEL_Markov
Bayesian and Markov-chain-based wind failure modeling for power transmission lines in Sicily using VAFFEL method.
# MSc Thesis Project – VAFFEL Method and Markov Chain Simulation (Sicily)
This repository contains the implementation of the VAFFEL methodology combined with 
Markov Chain modeling to estimate wind-induced failures in high-voltage power transmission 
lines across Sicily, Italy.

## Overview
The project integrates multiple geospatial and meteorological datasets, building a 
comprehensive spatial database of power transmission infrastructure, wind conditions, 
and failure events. The model applies Bayesian updating and stochastic simulations 
to estimate annual failure rates and fragility curves.

## Main Features
- Data extraction and merging from multiple sources (towers, circuits, wind datasets).
- Spatial matching of towers with 5×5 km weather grid cells.
- Bayesian updating of historical failure rates (Algorithm 2 – VAFFEL method).
- Markov Chain modeling of line state transitions (operational ↔ failed).
- Visualization of results and fragility distributions.

## Tools & Technologies
Python • Pandas • NumPy • GeoPandas • Matplotlib • NetCDF4 • VAFFEL • Bayesian Inference

## Author
**Samereh Gheibi**  
MSc Student – Politecnico di Torino  
Thesis Title: *Energy Informatics for Facilitating Large-Scale Research and Application*
Supervisor: Prof. Tao Huang

