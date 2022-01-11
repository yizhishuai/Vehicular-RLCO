# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:34:35 2021

@author: Mieszko Ferens
"""

import chainerrl
import gym
import time
import numpy as np

from agent_creator import make_training_agents
from graph_creator import (makeFigurePlot, makeFigureHistSingle,
                           makeFigureHistSubplot)

optimal_reward = 0 # Optimal reward of offloader

### - Computation offloading agents with ChainerRL - ###

if(__name__ == '__main__'):
    ## Environment (using gym)
    # Checking if the environment is already registered is necesary for
    # subsecuent executions
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'offload' in env:
            print('Remove {} from registry'.format(env))
            del gym.envs.registration.registry.env_specs[env]
    del env_dict
    
    env = gym.make('offloading_net:offload-v0')
    env = chainerrl.wrappers.CastObservationToFloat32(env)

# Function used to get training data on a specific scenario
def train_scenario(env):
    
    # Environment parameters requiered for training/testing
    n_actions = env.action_space.n
    n_apps = len(env.traffic_generator.apps)
    n_nodes = len(env.node_type) # TODO (This variable name is ok??)
    
    ## - To define the agent & training, will use ChainerRL - ##
    
    """
    To create various agent and compare them, many of the objects that the
    agent requieres have to be instanced multiple times. If not, they will
    share instances of certain objects and training will be biased.
    
    Also note that for comparing average Q values, gamma should be equal for
    all agents because this parameter influences their calculation.
    """
    
    # Discount factors
    gammas = 0.995
    
    # Algorithms to be used
    alg = ['DDQN', 'SARSA', 'PAL', 'TRPO']
    
    # Explorations that are to be analized (in algorithms that use them)
    explorators = 'const'
    epsilons = 0.2
    
    repetitions = 1
    
    """
    This function returns a list of lists. Each of the lists represents a type
    of agent and contains as many as repetitions declares (default is 1). Each
    of the internal lists contains turples that each contain all necesary
    objects for an agent to be trained.
    In the turples index 0 is the agent and index 1 is the info, the other
    indexes don't need to be accesed.
    Parameters:
        env: gym environment
        gammas: Discount factors to be used; you may define only one
        explorators: Exploration types to be used; you may define only one per
                     agent (constant, linear)
        epsilons: Epsilon values to be used; you may define only one per agent
        repetitions: How many replicas of a type of agent are to be created
                     (default is 1)
        alg: The algorithm of the agent; some of the other parameters don't
             matter depending on what algorithm is picked (DDQN, TRPO, SARSA,
             PAL) 
        NOTE 1: If you define multiple values for a parameter, you can only
                define one of the other if it is to be the same for all agent
                types
                Example 1: gammas = 0.7, alg = 'DDQN', explorators = 'const',
                           epsilons = [0.1, 0.2], repetitions = [2, 3]
                Example 2: gammas = [0.1, 0.5], alg = ['DDQN', 'SARSA'],
                           explorators = ['const', 'linear'],
                           epsilons = [0.2, [0.4,0.05,5000]], repetitions = 3
                Multiple value parameters are gammas, explorators, epsilons and
                repetitions
        NOTE 2: For algorithms with policy instead of a Q-function, explorators
                and epsilons parameters are unused (but need a value!)
                Algorithms with policy include: TRPO
    """
    agents = make_training_agents(
            env, gammas, explorators, epsilons, alg, repetitions)
    
    # Training
    """
    The training environment consists of a network in which the agent has to
    look for a path from an origin node to a destinition node. Each action will
    result in a negative reward which depends on the cost of traversing the
    link choosen by the agent.
    In some states (nodes), the agent can choose an invalid action due to there
    being more actions than outgoing links from the current node. These actions
    yield a significant cost and will not change the state of the environment.
    """
    print('---TRAINING---')
    # Number of time steps to archive a stationary state in the network
    start_up = 1000
    n_time_steps = 110000 # For 10^-3 precision -> ~10^5 sample points
    # Number of last episodes to use for average reward calculation
    averaging_window = 10000
    x_axis = range(1, start_up+n_time_steps+1) # X axis for ploting results
    # Stores average trining times for each type of agent
    average_total_training_times = []
    average_agent_training_times = []
    # Stores accumulated average rewards of best performing agent in each batch
    top_agents_average = []
    # Iterate through the different types of agents
    for batch in range(len(agents)):
        print('--Batch', batch, 'in training...')
        # Stores training times of agents of given type (of batch)
        total_training_times = []
        agent_training_times = []
        # Stores accumulated average rewards during training (all agents)
        average_reward_values = []
        # Iterate through the replicas of agents
        for a in range(len(agents[batch])):
            print('--Agent', a, 'in training...')
            # Stores rewards during training (one agent)
            rewards = []
            # Stores accumulated averaged rewards during training (one agent)
            average_rewards = []
            # Stores the time taken by the agent to process all time steps
            training_times = 0
            obs = env.reset() # Initialize environment for agent
            reward = 0 # Reward on time step
            done = False
            t = 0 # Time step
            time0 = time.time() # Training starts
            while not done and t < start_up + n_time_steps:
                time_agent = time.time() # Agent starts to process
                # Index 0 is agent object
                action = agents[batch][a][0].act_and_train(obs, reward)
                # Count the time the agents takes to process
                training_times += time.time() - time_agent
                obs, reward, done, _ = env.step(action) # Environment
                rewards.append(reward) # Store time step reward
                t += 1
                # Calculate and store the average reward after max time steps
                if(len(rewards) <= averaging_window):
                    average_rewards.append(
                        sum(rewards)/len(rewards))
                else: # Discard time steps older than averaging window length
                    average_rewards.append(
                            sum(rewards[t-averaging_window:t])
                            /averaging_window)
                # Show how training progresses
                if(__name__ == "__main__"):
                    if t % 10 == 0:
                        print('Time step', t)
                        if(reward >= 0):
                            print('Application processed succesfully')
                        else:
                            print('Application processed too slowly')
                        env.render()
            
            # End of training
            
            # Store elapsed time during training
            total_training_times.append(time.time() - time0)
            
            # Store elapsed time for agent's processing
            agent_training_times.append(training_times)
            
            # Store accumulated rewards of trained agent
            average_reward_values.append(average_rewards)
            
            # Look for the best performing agent
            if(a == 0):
                best = sum(average_reward_values[a])
                best_agent = a
            elif(best <= sum(average_reward_values[a])):
                best = sum(average_reward_values[a])
                best_agent = a
        
        # Store average training time of agent type (of batch)
        average_total_training_times.append(
            sum(total_training_times)/len(agents[batch]))
        
        # Store average processing time of agent type (of batch)
        average_agent_training_times.append(
            sum(agent_training_times)/len(agents[batch]))
        
        top_agents_average.append(average_reward_values[best_agent])
        
        """
        NOTE: For displaying agent information of a certain type (batch),
              second dimension index can be any existing agent (0 always works)
              and third dimension index 1 is agent info.
        """
        
        if(__name__ == "__main__"):
            # Plot results of batch (average rewards)
            labels = ['Time step', 'Average reward',
                      'Evolution of rewards (' + agents[batch][0][1] + ')']
            makeFigurePlot(
                x_axis, average_reward_values, optimal_reward, labels)
    
    if(__name__ == "__main__"):
        # Plot results of best performing agents (average rewards)
        labels = ['Time step', 'Average reward',
                  'Evolution of rewards (best agents)']
        legend = []
        for a in range(len(top_agents_average)):
            legend.append(agents[a][0][1])
        makeFigurePlot(
            x_axis, top_agents_average, optimal_reward, labels, legend)
    
    # Average times
    print("\n--Average agent processing times:")
    for batch in range(len(agents)):
        print(agents[batch][0][1], ': ', average_agent_training_times[batch],
              's', sep='')
    print("\n--Average training times:")
    for batch in range(len(agents)):
        print(agents[batch][0][1], ': ', average_total_training_times[batch],
              's', sep='')
    print('NOTE: The training time takes into account some data collecting!')
    
    # Testing
    # Testing average of successfully processed application for each batch
    test_success_rate = []
    # Testing average of proccessing distribution on nodes per application
    test_act_distribution = []
    # Testing average of total delay observed per application
    test_app_delay_avg = []
    # Testing total delays of each petition per application
    test_app_delays = []
    print('\n---TESTING---')
    n_time_steps = 100000
    for batch in range(len(agents)):
        batch_success_rate = []
        batch_act_distribution = []
        batch_app_count = []
        batch_app_processed = []
        batch_app_delay_avg = []
        batch_app_delays = []
        print(agents[batch][0][1], ':', sep='')
        for a in range(len(agents[batch])):
            print('  Replica', a, end=':\n')
            obs = env.reset()
            done = False
            reward = 0
            success = 0
            last_app = 0
            t = 0
            act_distribution = np.zeros((n_apps, n_nodes), dtype=np.float32)
            app_count = [0]*n_apps
            app_processed = [0]*n_apps
            app_delays = []
            for i in range(n_apps):
                app_delays.append([])
            while not done and t < start_up + n_time_steps:
                last_app = env.app
                action = agents[batch][a][0].act(obs)
                obs, reward, done, _ = env.step(action)
                if(t >= start_up):
                    if(reward >= 0):
                        success += 1
                        
                    # Count the times a certain node processed a specific app
                    act_distribution[last_app-1][action] += 1
                    app_count[last_app-1] += 1
                    
                    # Store the delay of the last processed application
                    if(env.total_delay >= 0): # Only if processed
                        app_delays[last_app-1].append(env.total_delay)
                        app_processed[last_app-1] += 1
                
                t += 1
            
            # Calculate the fraction of successfully processed applications
            batch_success_rate.append(success/n_time_steps)
            
            # Calculate the averages of application distribution throughout the
            # processing nodes
            for i in range(n_apps):
                act_distribution[i] = act_distribution[i]/app_count[i]
            batch_act_distribution.append(act_distribution)
            
            # Store the application petition count for the last simulation
            batch_app_count.append(app_count)
            
            # Store the processed application count for the last simulation
            batch_app_processed.append(app_processed)
            
            # Calculate and store the average total delay of each application
            temp = []
            for i in range(n_apps):
                temp.append(sum(app_delays[i])/app_processed[i])
            batch_app_delay_avg.append(temp)
            
            # Store the registered delays for the tested agents
            batch_app_delays.append(app_delays)
            
            # Print results of replica
            print('   -Success rate: ', batch_success_rate[a]*100, '%',
                  sep='')
            print('   -Processed application rate:')
            print('   |-> Apps: ', str(env.traffic_generator.apps), sep='')
            print('   |-> Rate: ', str(list(
                np.divide(batch_app_processed[a], batch_app_count[a]))),
                sep='')
            print('   -Action distribution:')
            for i in range(n_apps):
                print('   |-> App ', (i+1), ': ', sep='')
                print('    |-> Nodes: ', str(env.node_type) , sep='')
                print('    |-> Dist.: ',
                      str(batch_act_distribution[a][i]*100), '%', sep='')
            print('   -Total application delay average:')
            print('   |-> Apps:   ', str(env.traffic_generator.apps), sep='')
            print('   |-> Delays: ', str(batch_app_delay_avg[a]), sep='')
            
            """
            # Create graphs
            if(__name__ == "__main__"):
                # Create histogram of replica (app delay distribution)
                labels = ['Total application delay', '',
                          'Total application delay distribution ('
                          + agents[batch][0][1] + ' - Replica ' + str(a) + ')']
                legend = []
                for i in range(1, len(env.traffic_generator.apps) + 1):
                    legend.append('Application ' + str(i)) #Placeholder?
                bins = 64
                makeFigureHist(batch_app_delays[a], bins, labels, legend)
            """
        
        # Look for best performing agent (based on average delays)
        best = sum(batch_app_delay_avg[0])
        best_agent = 0
        for a in range(1, len(agents[batch])):
            temp = sum(batch_app_delay_avg[a])
            if(best > temp):
                best_agent = a
                best = temp
        
        # Store the average total delay per application of best agent of batch
        test_app_delay_avg.append(batch_app_delay_avg[best_agent])
        
        # Store the delay distribution per application of best agent of batch
        test_app_delays.append(batch_app_delays[best_agent])
        
        # Calculate the averages of successfully processed applications
        test_success_rate.append(
            sum(batch_success_rate)/len(batch_success_rate))
        
        # Store the averages of application distribution throughout the
        # processing nodes of best agent
        test_act_distribution.append(batch_act_distribution[best_agent])
        
        """
        # Create graphs
        if(__name__ == "__main__"):
            # Create histogram for best replica (app delay distribution)
            labels = ['Total application delay', '',
                      'Total application delay distribution ('
                      + agents[batch][0][1] + ' - Best' + ')']
            legend = []
            for i in range(1, n_apps + 1):
                legend.append('Application ' + str(i)) #Placeholder?
            bins = 10
            makeFigureHist(test_app_delays[batch], bins, labels, legend)
        """
    
    # Create histogram of delays of each application (only best agents)
    if(__name__ == "__main__"):
        for i in range(n_apps):
            labels = ['Total application delay', '',
                      env.traffic_generator.app_info[i]]
            legend = []
            y_axis = []
            for batch in range(len(test_act_distribution)):
                legend.append(agents[batch][0][1] + '(best)')
                y_axis.append(test_app_delays[batch][i])
            bins = 20
            max_delay = env.traffic_generator.app_max_delay[i]
            makeFigureHistSubplot(y_axis, bins, labels, legend, max_delay)
    
    """
    return {'train_block_probabilities': average_block_prob,
            'train_BW_block': average_BW_block,
            'train_voluntary_blocks': average_voluntary_blocks,
            'train_BW_voluntary_blocks': average_BW_voluntary_blocks,
            'train_benefit': average_benefit,
            'test_block_probabilities': test_block_prob,
            'test_BW_block': test_BW_block,
            'test_voluntary_blocks': test_voluntary_blocks,
            'test_BW_voluntary_blocks': test_BW_voluntary_blocks,
            'test_benefit': test_benefit,
            'test_shortest_path_similarity': test_shortest_path_similarity,
            'test_odu_voluntary_block_prob': test_odu_voluntary_block_prob,
            'existing_odu': existing_odu, 'agents': agents}
    """

    return {'agents':agents}

if(__name__ == "__main__"):
    train_scenario(env)

