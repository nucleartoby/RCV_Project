@echo off
IF EXIST monte_carlo_sim.exe (
    del monte_carlo_sim.exe
)
g++ mc_sim.cpp -o monte_carlo_sim
