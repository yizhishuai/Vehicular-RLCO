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
        #       it simply stores all petitions that have been generated
        #       randomly for the discrete event simulator
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
        application petition and generates a new one for the source that was
        the origin for that petition. This newly generated petition is inserted
        into the already generated petitions queue into the proper position
        depending on its arrival time
        """
        
        # The next generated petition (next in arrival time) to send to the
        # controller
        current_petition = self.petition_Q.pop(0)
        
        # Update arrival times for the rest of the petitions in the queue
        for i in range(len(self.petition_Q)):
            self.petition_Q[i][3] -= current_petition[3]
        
        ## Generate next petition from node that is about to be processed
        app_index = current_petition[0] - 1
        
        # Calculate the arrival time of the petition
        next_arrival_time = self.gen_distribution(
                self.app_rate[app_index], 'exponential')
        
        next_petition = [app_index + 1, current_petition[1],
                         current_petition[2], next_arrival_time,
                         self.app_cost[app_index], self.app_data_in[app_index],
                         self.app_data_out[app_index],
                         self.app_max_delay[app_index]]
        
        # Insert new petition in the corresponding queue position (according
        # to arrival time)
        i = next((x for x, f in enumerate(self.petition_Q)
                  if f[3] > next_arrival_time), -1)
        if(self.petition_Q and i >= 0):
            self.petition_Q.insert(i, next_petition)
        else:
            self.petition_Q.append(next_petition)
        
        # Return current petition to the controller
        return current_petition

    def gen_initial_traffic(self, node_vehicles):
        
        self.petition_Q.clear() # Clear the queue for initialization
        
        # Generate one petition for each application of each vehicle
        for node in range(self.net_nodes, self.n_nodes):
            for vehicle in range(node_vehicles):
                for app_index in range(len(self.apps)):
                    # Calculate the arrival time of the petition
                    next_arrival_time = self.gen_distribution(
                            self.app_rate[app_index], 'exponential')
                    
                    # Assign values to the queue (not sorted by arrival times)
                    self.petition_Q.append(
                        [app_index + 1, node + 1, vehicle, next_arrival_time,
                         self.app_cost[app_index], self.app_data_in[app_index],
                         self.app_data_out[app_index],
                         self.app_max_delay[app_index]])
        
        # Sort petitions by arrival time
        self.petition_Q.sort(key=lambda x:x[3])

    def gen_distribution(self, beta=1, dist='static'):
        if(dist in 'static'):
            return beta
        elif(dist in 'exponential'):
            return np.random.exponential(beta)
        else:
            raise KeyboardInterrupt('Unexpected type of distribution')

