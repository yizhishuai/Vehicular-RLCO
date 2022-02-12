# -*- coding: utf-8 -*-
"""
Created on Saturday Feb 12 11:08:57 2022

@author: Mieszko Ferens
"""

import random.randint

### Heuristic algorithms for load distributions

# TODO
"""
These classes implement the shortest path algorithm and k-shortest paths
algorithm for comparison with RL agents. They are designed to work in the
environments that tested RL agents are going to be working in.
"""

class local_processing_agent:
    
    """
    This agent always selects the last action, because in our environment that
    is always the local vehicle (processes the applications locally).
    """
    def __init__(self, n_actions):
        self.action = n_actions-1
    
    def act_and_train(self, obs, reward):
        # Input parameters are ignored since this is not a RL agent
        return self.action
    
    def act(self, obs):
        # Input parameter is ignored since this is not a RL agent
        return self.action

class cloud_processing_agent:
    
    """
    This agent always selects the first action, because in our environment that
    is always the cloud node (processes the applications at cloud).
    """
    def __init__(self):
        self.action = 0
    
    def act_and_train(self, obs, reward):
        # Input parameters are ignored since this is not a RL agent
        return self.action
    
    def act(self, obs):
        # Input parameter is ignored since this is not a RL agent
        return self.action

class uniform_distribution_agent:
    
    """
    This agent selects any of the possible action with uniform probability,
    distributing its decisions and the load equally.
    """
    def __init__(self, n_actions):
        self.last_action = n_actions-1
    
    def act_and_train(self, obs, reward):
        # Input parameters are ignored since this is not a RL agent
        return random.randint(0, self.last_action)
    
    def act(self, obs):
        # Input parameter is ignored since this is not a RL agent
        return random.randint(0, self.last_action)

class max_distance_agent:
    
    """
    This agent always selects the lowest value action possible given the
    total delay estimation to distribute the load in the network while
    attempting to reduce delays (the agent basically checks what node can
    process the application in theory going by the order of the furthest nodes
    first to the closest, i.e. cloud -> MECs -> RSUs -> local vehicle).
    It doesn't take into account the current load of the nodes.
    """
    def __init__(self, n_actions):
        self.action = n_actions-1
    
    def act_and_train(self, obs, reward):
        # Input parameters are ignored since this is not a RL agent
        return self.action
    
    def act(self, obs):
        # Input parameter is ignored since this is not a RL agent
        return self.action

# Funtion that instances shortest path algorithms imitating RL agents
def make_heuristic_agents(env):
    
    agent_local_processing = local_processing_agent(env.action_space.n)
    local_processing_info = 'Always local processing'
    
    agent_cloud_processing = cloud_processing_agent()
    cloud_processing_info = 'Always cloud processing'
    
    agent_uniform_distribution = uniform_distribution_agent(env.action_space.n)
    uniform_distribution_info = 'Uniform distribution of load'
    
    #agent_max_distance = max_distance_agent(env.action_space.n)
    #max_distance_info = 'Max distance processing'
    
    return [[agent_local_processing, local_processing_info],
            [agent_cloud_processing, cloud_processing_info],
            [agent_uniform_distribution, uniform_distribution_info]]

