# -*- coding: utf-8 -*-
"""
Created on Sat Nov  13 16:43:15 2021

@author: Mieszko Ferens
"""

import numpy as np

### Class for traffic generation

class traffic_generator():
    
    def __init__(self, n_nodes, net_nodes, apps, app_cost, app_data_in,
                 app_data_out, app_max_delay, app_rate, app_info):
        # The petition queue stores petitions that have been precalculated
        # NOTE: This queue does not represent pending petitions for the agent,
        #       it simply stores all petitions that have been generate randomly
        #       for the discrete event simulator
        self.petition_Q = []
        
        # Number of nodes in the network
        self.n_nodes = n_nodes
        # Number of non-vehicle nodes in the network
        self.net_nodes = net_nodes
        
        # Imported application parameters
        self.apps = apps
        self.app_cost = app_cost
        self.app_data_in = app_data_in
        self.app_data_out = app_data_out
        self.app_max_delay = app_max_delay
        self.app_info = app_info
        self.app_rate = app_rate
    
    def gen_traffic(self):
        
        """
        Every time this method is called it returns the next arriving
        application petition and generates a new one for the node that was the
        origin for that petition. This newly generated petition is inserted
        into the already generated petitions queue into the proper position
        depending on its arrival time
        """
        
        # Send to the controller the next generated petition (next in arrival
        # time)
        current_petition = self.petition_Q.pop(0)
        
        # Update arrival times for the rest of the petitions in the queue
        for i in range(len(self.petition_Q)):
            self.petition_Q[i][2] -= current_petition[2]
        
        ## Generate next petition from node that is about to be processed
        
        # Calculate the arrival time of the petition
        next_arrival_time = self.gen_distribution(
                self.app_rate[current_petition[1]-1], 'exponential')
        # Randomly select an application
        app_index = np.random.choice(self.apps) - 1
        
        next_petition = [app_index + 1, current_petition[1], next_arrival_time,
                         self.app_cost[app_index], self.app_data_in[app_index],
                         self.app_data_out[app_index],
                         self.app_max_delay[app_index]]
        
        # Insert new petition in the corresponding queue position (according
        # to arrival time)
        for i in range(len(self.petition_Q)):
            if(next_petition[2] < self.petition_Q[i][2]):
                self.petition_Q.insert(i, next_petition)
                break
            if(i == len(self.petition_Q) - 1):
                self.petition_Q.append(next_petition)
        
        # Return current petition to the controller
        return current_petition

    def gen_initial_traffic(self):
        
        self.petition_Q.clear() # Clear the queue for initialization
        
        # Count amount of nodes that will generate traffic
        loop_nodes = max(2, self.n_nodes - self.net_nodes)
        # NOTE: The way gen_traffic() works it requieres at least 2 petitions
        # in the queue
        
        # Generate one petition from each node
        initial_petitions = []
        for i in range(self.net_nodes, self.net_nodes + loop_nodes):
            # Calculate the arrival time of the petition
            next_arrival_time = self.gen_distribution(
                    self.app_rate[i], 'exponential')
            # Randomly select an application
            app_index = np.random.choice(self.apps) - 1
            
            # Assign values to the temporary queue (not sorted by arrival times)
            initial_petitions.append(
                [app_index + 1, i + 1, next_arrival_time, self.app_cost[app_index],
                 self.app_data_in[app_index], self.app_data_out[app_index],
                 self.app_max_delay[app_index]])
        
        # In case there was only one node but the minimum of 2 petitions were
        # generated, change the origin of the second one
        if(self.n_nodes - self.net_nodes < 2):
            initial_petitions[1][1] = initial_petitions[0][1]
        
        # Sort petitions by arrival time
        for index, f in enumerate(initial_petitions):
            if(len(self.petition_Q) == 0): # If the queue is empty
                self.petition_Q.append(f)
            else: # If the queue is not empty
                for i in range(len(self.petition_Q)):
                    if(f[2] < self.petition_Q[i][2]):
                        self.petition_Q.insert(i, f)
                        break
                    if(i == len(self.petition_Q) - 1):
                        self.petition_Q.append(f)
        
        # All arrival times are trimmed so that the first arriving petition has
        # a next arrival time of 0
        first_arrival_time = self.petition_Q[0][2]
        for i in range(len(self.petition_Q)):
            self.petition_Q[i][2] -= first_arrival_time

    def gen_distribution(self, beta=1, dist='static'):
        if(dist in 'static'):
            return beta
        elif(dist in 'exponential'):
            return np.random.exponential(beta)
        else:
            print('Unexpected type of distribution')
            raise KeyboardInterrupt

