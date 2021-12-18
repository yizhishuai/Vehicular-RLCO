# -*- coding: utf-8 -*-
"""
Created on Sat Nov  13 16:44:00 2021

@author: Mieszko Ferens
"""

### Environment for computation offloading

import numpy as np
import gym
from gym import spaces

from .traffic_generator import traffic_generator
from .core_manager import core_manager
from .parameters import links, links_rate, links_delay, node_type, node_clock, node_cores, n_nodes, net_nodes, all_paths, node_comb, apps, app_max_delay, app_info

"""
Explanation on implemented discrete event simulator:
    TODO

In this environment the petitions are generated in the traffic_generator class
and passed to the environment.
"""

class offload_netEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):   
        # Node type list (used as information for metrics during testing)
        self.node_type = node_type
        # App type list (used as information for metrics during testing)
        self.apps = apps
        # App description list (used as information for metrics during testing)
        self.app_info = app_info
        # Maximum tolerable delay of each application
        self.apps_max_delay = app_max_delay
        
        # Type of next application
        self.app = 0
        # Origin node of next application (vehicle)
        self.app_origin = 0
        # Time until next application
        self.app_time = 0
        # Cost of processing the next application (clock clycles/bit)
        self.app_cost = 0
        # Input data quantity from the next application (bits)
        self.app_data_in = 0
        # Output data quantity from the next application (bits)
        self.app_data_out = 0
        # Maximum tolerable delay for the next application (ms)
        self.app_max_delay = 0
        
        # Total delay of current application
        self.total_delay = 0
        
        # For monitoring purposes the observation will be kept at all times
        self.obs = 0
        
        # List that holds individual core info for all nodes (except cloud)
        #self.cores = []
        # Number of cores in the network (except cloud)
        self.n_cores = 0
        for a in range(n_nodes):
            if(node_cores[a] > 0): # Ignore cloud nodes
                self.n_cores += node_cores[a]
        
        # Discrete event simulator traffic generation initialization
        self.traffic_generator = traffic_generator()
        
        # Discrete event simulator core manager initialization
        self.core_manager = core_manager()
        
        # The observation space has an element per core in the network
        self.observation_space = spaces.Box(low=0, high=1, shape=(
                self.n_cores + len(self.apps), 1), dtype=np.float32)
        
        self.action_space = spaces.Discrete(net_nodes + 1)

    def step(self, action):
        
        # For each application, the node that processes it cannot be another
        # vehicle, but the local vehicle (the one that generates the petition)
        # can
        # Translate the action to the correct node number if the agent decides
        # to processes the application locally
        if(action == net_nodes):
            action = self.app_origin - 1
            path = []
        # Prepare path between origin node and the selected processor
        else:
            current_nodes = [self.app_origin, action + 1]
            current_nodes.sort()
            path = all_paths[node_comb.index(tuple(current_nodes))]
        
        # For each action one node for processing an application is chosen,
        # reserving one of its cores (might queue)
        
        # Calculate the transmission delay for the application's data
        forward_delay = 0
        return_delay = 0
        if path:
            for a in range(len(path[0])-1):
                link = [path[0][a], path[0][a+1]]
                link.sort()
                link_index = links.index(link)
                link_rate = links_rate[link_index] # in Mbit/s
                link_delay = links_delay[link_index] # in ms
                
                forward_delay += (self.app_data_in/link_rate) + link_delay
                return_delay += link_delay + (self.app_data_out/link_rate)
        
        # Calculate the processing delay at the node
        proc_delay = (self.app_data_in*self.app_cost/node_clock[action])
        
        # Choose the next as soon to be available core from the selected
        # processing node and reserve its core for the required time
        # If the selected processing node is the cloud no reservation is done
        # as it has infinite resources
        self.total_delay = 0
        if(action != 0): # Not cloud
            self.total_delay = self.core_manager.reserve(
                action-1, forward_delay, proc_delay, return_delay)
        else: # Cloud
            # Calculate the total delay of the application processing
            self.total_delay = forward_delay + proc_delay + return_delay
        
        # Check estimated delay for the application and calculate reward
        if(self.total_delay > 0): # Application processed
            reward = min(0, self.app_max_delay - self.total_delay)
        else: # Application not processed
            reward = -1000
        
        # Get next arriving petition
        next_petition = self.traffic_generator.gen_traffic()
        # Assign variables from petition
        self.app = next_petition[0]
        self.app_origin = next_petition[1]
        self.app_time = next_petition[2]
        self.app_cost = next_petition[3]
        self.app_data_in = next_petition[4]
        self.app_data_out = next_petition[5]
        self.app_max_delay = next_petition[6]
        
        ## Observation calculation
        # Core information
        self.obs = self.core_manager.update_and_calc_obs(self.app_time)
        
        # Add current petition's application type to observation
        app_type = [0]*len(self.apps)
        app_type[self.app-1] = 1
        self.obs = np.append(self.obs, np.array(app_type, dtype=np.float32))

        done = False # This environment is continuous and is never done

        return np.array([self.obs, reward, done, ""])

    def reset(self):
        
        # Reset all cores
        self.core_manager.reset(n_nodes, node_cores)
        
        # Generate initial petitions to get things going
        self.traffic_generator.gen_initial_traffic()
        
        # Get first petition
        next_petition = self.traffic_generator.gen_traffic()
        # Assign variables from petition
        self.app = next_petition[0]
        self.app_origin = next_petition[1]
        self.app_time = next_petition[2]
        self.app_cost = next_petition[3]
        self.app_data_in = next_petition[4]
        self.app_data_out = next_petition[5]
        self.app_max_delay = next_petition[6]
        
        # Calculate observation
        self.obs = np.array(
            [0]*(self.n_cores + len(self.apps)), dtype=np.float32)
        
        return self.obs

    def render(self, mode='human'):
        # Print current core reservation times
        print('Core reservation time:', self.obs[0:self.n_cores])
        # Print next application to be processed
        print('Next application:', self.app)

