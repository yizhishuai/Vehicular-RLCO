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

from offloader import define_scenario, train_scenario, test_scenario
from graph_creator import makeFigurePlot

def parametric_sim_vehicles_train_per_test(env, topology, n_vehicles):
    
    # Parameter error
    if(not isinstance(n_vehicles, list)):
        raise KeyboardInterrupt("The vehicle variation for the simulation must"
                                " be defined as a list.\nTIP: Check the passed"
                                " parameter.")
    
    ## Run simulations with varying network load, training the agents for each
    ## network scenario
    
    # Metrics
    train_avg_total_times = []
    train_avg_agent_times = []
    test_success_rate = []
    
    # Train and test the agents
    for i in range(len(n_vehicles)):
        # Vary network load parameters
        env.set_total_vehicles(n_vehicles[i])
        
        # Get metrics of trained and tested agents
        agents = define_scenario(env) # Get base agents
        train_results = train_scenario(env, agents)
        test_results = test_scenario(env, agents)
        train_avg_total_times.append(train_results['train_avg_total_times'])
        train_avg_agent_times.append(train_results['train_avg_agent_times'])
        test_success_rate.append(test_results['test_success_rate'])
    
    ## Plot results
    
    # Reshape data to plot with makeFigurePlot function
    train_avg_total_times = reshape_data(train_avg_total_times)
    train_avg_agent_times = reshape_data(train_avg_agent_times)
    test_success_rate = reshape_data(test_success_rate)
    """
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
    """
    
    # Plot graphs
    labels = ['Vehicles in network', 'Training average total times',
              topology]
    legend = []
    for a in range(len(agents)):
        legend.append(agents[a][0][1])
        
    makeFigurePlot(
        n_vehicles, train_avg_total_times, labels=labels, legend=legend)
    plt.savefig('Figures/' + labels[1] + '.svg')
    labels[1] = 'Training average agent processing times'
    makeFigurePlot(
        n_vehicles, train_avg_agent_times, labels=labels, legend=legend)
    plt.savefig('Figures/' + labels[1] + '.svg')
    labels[1] = 'Testing success rate'
    makeFigurePlot(
        n_vehicles, test_success_rate, labels=labels, legend=legend)
    plt.savefig('Figures/' + labels[1] + '.svg')
    
    plt.close('all') # Close all figures
    
    ## Log the data of the experiment in a file
    
    # Open log file
    try:
        log_file = open("TestLog_" + str(date.today()) + '.txt', 'wt',
                        encoding='utf-8')
    except:
        raise KeyboardInterrupt('Error while initializing log file...aborting')
    
    # Initial information
    log_file.write("Experiment Log - " + str(datetime.today()) + '\n\n')
    log_file.write("Network topology: " + topology + '\n')
    
    log_file.write("---------------------------------------------------\n\n")
    
    # Data
    log_file.write('n_vehicles = ' + str(n_vehicles) + '\n')
    for a in range(len(agents)):
        log_file.write("\n---" + agents[a][0][1] + '\n')
        log_file.write("-Training average total times:\n" +
                       str(train_avg_total_times[a]) + '\n')
        log_file.write("-Training average agent processing times:\n" +
                       str(train_avg_agent_times[a]) + '\n')
        log_file.write("-Test success rate:\n" + str(test_success_rate[a]) +
                       '\n')
    
    log_file.write("---------------------------------------------------\n\n")
    # .csv
    log_file.write("n_vehicles")
    for a in range(len(agents)):
        for key in train_results.keys():
            if(key != 'agents'):
                log_file.write(',' + key + '-' + agents[a][0][1])
        for key in test_results.keys():
            log_file.write(',' + key + '-' + agents[a][0][1])
    for i in range(len(n_vehicles)):
        log_file.write('\n' + str(n_vehicles[i]) + ',')
        for a in range(len(agents)):
            log_file.write(str(train_avg_total_times[a][i]) + ',')
            log_file.write(str(train_avg_agent_times[a][i]) + ',')
            log_file.write(str(test_success_rate[a][i]) + ',')
    
    log_file.close() # Close log file

def parametric_sim_errorVar_train_per_test(env, topology, estimation_err_var):
    
    # Parameter error
    if(not isinstance(estimation_err_var, list)):
        raise KeyboardInterrupt("The estimation error variance variation for "
                                "the simulation must be defined as a list.\n"
                                "TIP: Check the passed parameter.")
    
    ## Run simulations with varying network load, training the agents for each
    ## network scenario
    
    # Metrics
    train_avg_total_times = []
    train_avg_agent_times = []
    test_success_rate = []
    
    # Train and test the agents
    for i in range(len(estimation_err_var)):
        # Vary network load parameters
        env.set_error_var(estimation_err_var[i])
        
        # Get metrics of trained and tested agents
        agents = define_scenario(env) # Get base agents
        train_results = train_scenario(env, agents)
        test_results = test_scenario(env, agents)
        train_avg_total_times.append(train_results['train_avg_total_times'])
        train_avg_agent_times.append(train_results['train_avg_agent_times'])
        test_success_rate.append(test_results['test_success_rate'])
    
    ## Plot results
    
    # Reshape data to plot with makeFigurePlot function
    train_avg_total_times = reshape_data(train_avg_total_times)
    train_avg_agent_times = reshape_data(train_avg_agent_times)
    test_success_rate = reshape_data(test_success_rate)
    """
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
    """
    
    # Plot graphs
    labels = ['Processing time error variance', 'Training average total times',
              topology]
    legend = []
    for a in range(len(agents)):
        legend.append(agents[a][0][1])
        
    makeFigurePlot(estimation_err_var, train_avg_total_times, labels=labels,
                   legend=legend)
    plt.savefig('Figures/' + labels[1] + '.svg')
    labels[1] = 'Training average agent processing times'
    makeFigurePlot(estimation_err_var, train_avg_agent_times, labels=labels,
                   legend=legend)
    plt.savefig('Figures/' + labels[1] + '.svg')
    labels[1] = 'Testing success rate'
    makeFigurePlot(estimation_err_var, test_success_rate, labels=labels,
                   legend=legend)
    plt.savefig('Figures/' + labels[1] + '.svg')
    
    plt.close('all') # Close all figures
    
    ## Log the data of the experiment in a file
    
    # Open log file
    try:
        log_file = open("TestLog_" + str(date.today()) + '.txt', 'wt',
                        encoding='utf-8')
    except:
        raise KeyboardInterrupt('Error while initializing log file...aborting')
    
    # Initial information
    log_file.write("Experiment Log - " + str(datetime.today()) + '\n\n')
    log_file.write("Network topology: " + topology + '\n')
    
    log_file.write("---------------------------------------------------\n\n")
    
    # Data
    log_file.write('estimation_err_var = ' + str(estimation_err_var) + '\n')
    for a in range(len(agents)):
        log_file.write("\n---" + agents[a][0][1] + '\n')
        log_file.write("-Training average total times:\n" +
                       str(train_avg_total_times[a]) + '\n')
        log_file.write("-Training average agent processing times:\n" +
                       str(train_avg_agent_times[a]) + '\n')
        log_file.write("-Test success rate:\n" + str(test_success_rate[a]) +
                       '\n')
    
    log_file.write("---------------------------------------------------\n\n")
    # .csv
    log_file.write("estimation_err_var")
    for a in range(len(agents)):
        for key in train_results.keys():
            if(key != 'agents'):
                log_file.write(',' + key + '-' + agents[a][0][1])
        for key in test_results.keys():
            log_file.write(',' + key + '-' + agents[a][0][1])
    for i in range(len(estimation_err_var)):
        log_file.write('\n' + str(estimation_err_var[i]) + ',')
        for a in range(len(agents)):
            log_file.write(str(train_avg_total_times[a][i]) + ',')
            log_file.write(str(train_avg_agent_times[a][i]) + ',')
            log_file.write(str(test_success_rate[a][i]) + ',')
    
    log_file.close() # Close log file

# Funtion for reshaping the parametric simulation's results
def reshape_data(data):
    temp = np.array(data)
    reshaped_data = []
    for i in range(len(data[0])):
        reshaped_data.append(list(temp[:,i]))
    
    return reshaped_data

if(__name__ == "__main__"):
    
    path_to_env = "../Environments/offloading-net/offloading_net/envs/"
    
    ## Setup simulation state in temporal file (used for creating the
    ## appropriate environment, i.e. using the correct network topology)
    
    topologies = ["network_branchless"]
    topology_labels = ["Branchless network"]
    
    top_index = 0 # Pick one index of the above
    
    ## Define what is the current network topology for simulation in state file
    try:
        state_file = open(path_to_env + "net_topology", 'wt')
    except:
        raise KeyboardInterrupt(
            'Error while initializing state file...aborting')
    
    state_file.write(topologies[top_index])
    state_file.close()
    
    # Checking if the environment is already registered is necessary for
    # subsecuent executions
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for envs in env_dict:
        if 'offload' in envs:
            print('Remove {} from registry'.format(envs))
            del gym.envs.registration.registry.env_specs[envs]
    del env_dict
    
    ## Environment (using gym)
    env = gym.make('offloading_net:offload-v0')
    env = chainerrl.wrappers.CastObservationToFloat32(env)
    
    # Remove temporary file so it's not read in other simulations
    if(os.path.exists(path_to_env + "net_topology")):
        os.remove(path_to_env + "net_topology")
    
    # Simulation parameters
    n_vehicles = [10, 30, 50, 70, 90]
    estimation_err_var = [0, 1, 2, 3, 4]
    
    # Run simulations # TODO
    parametric_sim_vehicles_train_per_test(
        env, topology_labels[top_index], n_vehicles)
    
    #parametric_sim_errorVar_train_per_test(
    #    env, topology_labels[top_index], estimation_err_var)

