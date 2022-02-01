# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:51:16 2021

@author: Mieszko Ferens
"""

### Script for parametric simulations ###
"""
This script's use is to extract data from simulations with different parameters
in the environment so as to get metrics spanning various scenarios
"""

import os
import numpy as np
import gym
import chainerrl
import matplotlib.pyplot as plt
from datetime import date, datetime

from offloader import train_scenario
from graph_creator import makeFigurePlot

## Setup simulation state in temporal file (used for creating the appropriate
## environment, i.e. using the correct network topology)

path_to_env = "../Environments/offloading-net/offloading_net/envs/"

topologies = ["network_branchless"]
topology_labels = ["Branchless network"]

n_sim = 0 # Pick one index of the above

# Checking if the environment is already registered is necessary for subsecuent 
# executions
env_dict = gym.envs.registration.registry.env_specs.copy()
for envs in env_dict:
    if 'offload' in envs:
        print('Remove {} from registry'.format(envs))
        del gym.envs.registration.registry.env_specs[envs]
del env_dict

## Define what is the current network topology for simulation in state file
try:
    state_file = open(path_to_env + "net_topology", 'wt')
except:
    raise KeyboardInterrupt('Error while initializing state file...aborting')

state_file.write(topologies[n_sim])
state_file.close()

## Environment (using gym)
env = gym.make('offloading_net:offload-v0')
env = chainerrl.wrappers.CastObservationToFloat32(env)

## Run simulations with varying network load on current network

train_avg_total_times = []
train_avg_agent_times = []
test_success_rate = []
n_vehicles = []
for i in range(5):
    # Vary network load parameters
    n_vehicles.append(100 + 200*i)
    env.set_total_vehicles(n_vehicles[i])
    
    # Get metrics of trained and tested agents
    results = train_scenario(env)
    train_avg_total_times.append(results['train_avg_total_times'])
    train_avg_agent_times.append(results['train_avg_agent_times'])
    test_success_rate.append(results['test_success_rate'])

agents = results['agents']

## Plot results

# Reshape data to plot with makeFigurePlot function
temp1 = np.array(train_avg_total_times)
temp2 = np.array(train_avg_agent_times)
temp3 = np.array(test_success_rate)
n_plots = len(test_success_rate[0])
train_avg_total_times.clear()
train_avg_agent_times.clear()
test_success_rate.clear()
for i in range(n_plots):
    train_avg_total_times.append(list(temp1[:,i]))
    train_avg_agent_times.append(list(temp2[:,i]))
    test_success_rate.append(list(temp3[:,i]))
del temp1, temp2, temp3

# Plot graphs
labels = ['Vehicles in network', 'Training average total times',
          topology_labels[n_sim]]
legend = []
for a in range(len(agents)):
    legend.append(agents[a][0][1])
    
makeFigurePlot(n_vehicles, train_avg_total_times, labels=labels, legend=legend)
plt.savefig('Figures/' + labels[1] + '.svg')
labels[1] = 'Training average agent processing times'
makeFigurePlot(n_vehicles, train_avg_agent_times, labels=labels, legend=legend)
plt.savefig('Figures/' + labels[1] + '.svg')
labels[1] = 'Testing success rate'
makeFigurePlot(n_vehicles, test_success_rate, labels=labels, legend=legend)
plt.savefig('Figures/' + labels[1] + '.svg')

plt.close('all') # Close all figures

## Log the data of the experiment in a file

env.reset() # Reset the environment to get initial link capacities

# Open log file
try:
    log_file = open("TestLog_" + str(date.today()) + '.txt', 'wt',
                    encoding='utf-8')
except:
    raise KeyboardInterrupt('Error while initializing log file...aborting')

# Initial information
log_file.write("Experiment Log - " + str(datetime.today()) + '\n\n')
log_file.write("Network topology: " + topology_labels[n_sim] + '\n')

log_file.write("---------------------------------------------------\n\n")

# Data
log_file.write('n_vehicles = ' + str(n_vehicles) + '\n')
for a in range(len(agents)):
    log_file.write("\n---" + agents[a][0][1] + '\n')
    log_file.write("-Training average total times:\n" +
                   str(train_avg_total_times[a]) + '\n')
    log_file.write("-Training average agent processing times:\n" +
                   str(train_avg_agent_times[a]) + '\n')
    log_file.write("-Test success rate:\n" + str(test_success_rate[a]) + '\n')

log_file.write("---------------------------------------------------\n\n")
# .csv
log_file.write("n_vehicles")
for a in range(len(agents)):
    for key in results.keys():
        log_file.write(',' + key + '-' + agents[a][0][1])
for i in range(len(n_vehicles)):
    log_file.write('\n' + str(n_vehicles[i]))
    for a in range(len(agents)):
        log_file.write(str(train_avg_total_times[a][i]) + ',')
        log_file.write(str(train_avg_agent_times[a][i]) + ',')
        log_file.write(str(test_success_rate[a][i]) + ',')

log_file.close() # Close log file

# Finished (remove temporary file so it's not read in other simulations)
if(os.path.exists(path_to_env + "net_topology")):
    os.remove(path_to_env + "net_topology")

