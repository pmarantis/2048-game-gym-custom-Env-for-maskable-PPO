# 2048-game-gym-custom-Env-for-maskable-PPO
This repo includes a basic version of the 2048 game and a custom open ai gym env using the Maskable PPO algorithm from stablebaselines3contrib.
The states are log2 representations of the values of the game board and the reward is the log2 sum of merged tiles for the performed action.
Credits to https://www.geeksforgeeks.org/2048-game-in-python/ for the basic code for the game, which i altered to fit this project. 
The purpose is to show an implementation of MaskablePPO on a custom gym env with invalid moves. A basic trained model is included, trained with
the default MaskablePPO for a few million episodes, which only achieves 1024 around 15% of the time. 
Any insight on how the performance can be improved is appreciated. Feel free to play around with the code, application of CNNPolicy with a more suited
net architecture or a better reward function could possibly improve the training results.
