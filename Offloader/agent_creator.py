# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:27:40 2020

@author: Mieszko Ferens
"""

import chainer
import chainerrl
import chainer.links as L
import chainer.functions as F

# Q-function definition
class QFunction(chainer.Chain):
    
    def __init__(self, obs_size, n_actions, n_hidden_channels=60):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_actions)
    
    def __call__(self, obs, test=False):
        h = F.tanh(self.l0(obs))
        return chainerrl.action_value.DiscreteActionValue(self.l1(h))

# Funtion that instances an agent with certain parameters
def create_agent(
        gamma, obs_size, n_actions, exploration_func, alg, exp_type='constant',
        epsilon=0.1):
    
    # Error handling
    if(type(gamma) != float and type(gamma) != int):
        raise KeyboardInterrupt(
            'Error while creating agent: Gamma type invalid')
    if(type(exp_type) != str):
        raise KeyboardInterrupt(
            'Error while creating agent: Exploration type invalid')
    if(type(epsilon) != float and type(epsilon) != int
       and type(epsilon) != list):
        raise KeyboardInterrupt(
            'Error while creating agent: Epsilon type invalid')
    
    if(alg in 'DDQN'):
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('DDQN: ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('DDQN: ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000000)
        
        # Agent (DDQN)
        agent = chainerrl.agents.DoubleDQN(
                q_func, opt, replay_buffer, gamma, explorer,
                replay_start_size=100000, target_update_interval=50000)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'SARSA'):
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('SARSA: ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('SARSA: ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        
        # Agent (SARSA)
        agent = chainerrl.agents.SARSA(
                q_func, opt, replay_buffer, gamma, explorer,
                replay_start_size=100000, target_update_interval=50000)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'PAL'):
        # Q function instanciation
        q_func = QFunction(obs_size, n_actions)
        
        # Optimizer
        opt = chainer.optimizers.Adam(eps=1e-2)
        opt.setup(q_func)
        
        # Exploration & agent info
        if(exp_type in 'constant'):
            agent_info = ('PAL: ' + 'Constant ' + chr(949) + '=' +
                          str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                          ')')
            explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    epsilon=epsilon, random_action_func=exploration_func)
        elif(exp_type in 'linear decay'):
            if(type(epsilon) != list):
                raise KeyboardInterrupt(
                    'Error while creating agent: Linear decay exploration does'
                    'not work with a constant epsilon')
            agent_info = ('PAL: ' + 'Linear decay ' + chr(949) + '=' +
                          str(epsilon[0]) + '->' + str(epsilon[1]) + ' in ' +
                          str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                          '=' + str(gamma) + ')')
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=epsilon[0], end_epsilon=epsilon[1],
            decay_steps=epsilon[2], random_action_func=exploration_func)
        else: # If type doesn't match with any known one raise error
            raise KeyboardInterrupt(
                'Error while creating agent: Unknown exploration type')
        
        # Experience replay
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        
        # Agent (PAL)
        agent = chainerrl.agents.PAL(
                q_func, opt, replay_buffer, gamma, explorer,
                replay_start_size=100000, target_update_interval=50000)
        
        return agent, agent_info, q_func, opt, explorer, replay_buffer
    elif(alg in 'TRPO'):
        # Policy
        policy = chainerrl.policies.FCSoftmaxPolicy(
                obs_size, n_actions, n_hidden_channels=60, n_hidden_layers=1,
                last_wscale=0.01, nonlinearity=F.tanh)
        
        # Value function
        vf = chainerrl.v_functions.FCVFunction(
                obs_size, n_hidden_channels=60, n_hidden_layers=1,
                last_wscale=0.01, nonlinearity=F.tanh)
        
        # Optimizer
        opt = chainer.optimizers.Adam()
        opt.setup(vf)
        
        # Agent (TRPO)
        agent = chainerrl.agents.TRPO(
                policy=policy, vf=vf, vf_optimizer=opt, gamma=gamma,
                update_interval=50000)
        
        # Info
        agent_info = 'TRPO' + ' (' + chr(947) + '=' + str(gamma) + ')'
        
        return agent, agent_info, policy, opt

# Funtion that instances one or more agents with certain parameters
def create_agents(
        exp_type, epsilon, repetitions, gamma, obs_size, n_actions,
        exploration_func, alg):
    
    agents = []
    for i in range(repetitions):
        # Instance agent with its objects and append to list
        agents.append(create_agent(gamma, obs_size, n_actions,
                                   exploration_func, alg, exp_type, epsilon))
    
    # Return list of turples which contain agents with equal parameters
    return agents

# Funtion that creates instances of certain agents based on parameters
def make_training_agents(
        env, gammas, exp_types, epsilons, alg, repetitions=1):
    
    # If there are multiple algorithms process is repeated for each one
    if(type(alg) != list and type(alg) == str):
        alg = [alg]
    elif(type(alg) != list and type(alg) != str):
        raise KeyboardInterrupt(
            'Error while creating agents: Incorrect parameter types')
    
    agents = [] # List that contains lists of turples
    for a in range(len(alg)):
        
        # Check and save unused parameters for current algorithm
        if(alg[a] in 'TRPO'):
            exp_types_save = exp_types
            exp_types = 'const' # Any valid value
            epsilons_save = epsilons
            epsilons = 0.2 # Any valid value
            print('Warning: Not using exp_types and epsilons parameters for' +
                  alg[a])
        
        # Error handling
        if(exp_types == None or epsilons == None or gammas == None):
            raise KeyboardInterrupt(
                'Error while creating agents: Missing required arguments')
        if((type(exp_types) != str and type(exp_types) != list) or
           (type(epsilons) != float and type(epsilons) != list and
            type(epsilons) != int) or
           (type(repetitions) != int and type(repetitions) != list) or
           (type(gammas) != float and type(gammas) != list and
            type(gammas) != int)):
               raise KeyboardInterrupt(
                   'Error while creating agents: Incorrect parameter types')
        
        # Define some parameters from the environment
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n
        exploration_func = env.action_space.sample
        
        # Parameter check
        if(type(exp_types) == str and type(epsilons) == float and
           (type(gammas) == float or type(gammas) == int)):
            # One type of agent
            # Error handling
            if(type(repetitions) == list):
                raise KeyboardInterrupt(
                    'Error while creating agents: Repetitions parameter was '
                    'expected to be an integer')
            # Agent instanciation
            agents.append(create_agents(
                exp_types, epsilons, repetitions, gammas, obs_size, n_actions,
                exploration_func, alg[a]))
        
        elif(type(exp_types) == list):
            # Multiple types of agent
            agent_num = len(exp_types)
            
            # Error handling
            if(type(epsilons) == list and len(epsilons) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of epsilons\n'
                    'TIP: You may pass just one value of epsilon')
            if(type(repetitions) == list and len(repetitions) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of repetitions\n'
                    'TIP: You may pass just one value of repetitions')
            if(type(gammas) == list and len(gammas) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of gammas\n'
                    'TIP: You may pass just one value of gamma')
            
            # Agent instanciation
            
            # Create a list of just one element in case there is only one, so
            # code in the loop can be simplified (epsilons[i], repetitions[i],
            # gammas[i])
            if(type(epsilons) == float):
                temp = []
                for i in range(agent_num):
                    temp.append(epsilons)
                epsilons = temp
            if(type(repetitions) == int):
                temp = []
                for i in range(agent_num):
                    temp.append(repetitions)
                repetitions = temp
            if(type(gammas) == float or type(gammas) == int):
                temp = []
                for i in range(agent_num):
                    temp.append(gammas)
                gammas = temp
            
            for i in range(agent_num):
                """
                Create one list of turples, each turple contains data on an
                instance of an agent. The list contains agents with equal
                parameters.
                """
                agents.append(create_agents(
                    exp_types[i], epsilons[i], repetitions[i], gammas[i],
                    obs_size, n_actions, exploration_func, alg[a]))
        
        elif(type(epsilons) == list):
            # Multiple types of agent
            agent_num = len(epsilons)
            
            # Error handling
            if(type(exp_types) == list and len(exp_types) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of epsilons\n'
                    'TIP: You may pass just one value of epsilon')
            if(type(repetitions) == list and len(repetitions) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of repetitions\n'
                    'TIP: You may pass just one value of repetitions')
            if(type(gammas) == list and len(gammas) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of gammas\n'
                    'TIP: You may pass just one value of gamma')
            
            # Agent instanciation
            
            # Create a list of just one element in case there is only one, so
            # code in the loop can be simplified (exp_types[i], repetitions[i],
            # gammas[i])
            if(type(exp_types) == str):
                temp = []
                for i in range(agent_num):
                    temp.append(exp_types)
                exp_types = temp
            if(type(repetitions) == int):
                temp = []
                for i in range(agent_num):
                    temp.append(repetitions)
                repetitions = temp
            if(type(gammas) == float or type(gammas) == int):
                temp = []
                for i in range(agent_num):
                    temp.append(gammas)
                gammas = temp
            
            for i in range(agent_num):
                """
                Create one list of turples, each turple contains data on an
                instance of an agent. The list contains agents with equal
                parameters.
                """
                agents.append(create_agents(
                    exp_types[i], epsilons[i], repetitions[i], gammas[i],
                    obs_size, n_actions, exploration_func, alg[a]))
        
        elif(type(gammas) == list):
            # Multiple types of agent
            agent_num = len(gammas)
            
            # Error handling
            if(type(epsilons) == list and len(epsilons) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of epsilons\n'
                    'TIP: You may pass just one value of epsilon')
            if(type(exp_types) == list and len(exp_types) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of epsilons\n'
                    'TIP: You may pass just one value of epsilon')
            if(type(repetitions) == list and len(repetitions) != agent_num):
                raise KeyboardInterrupt(
                    'Error while creating agents: Number of exploration types '
                    'doesn´t match number of repetitions\n'
                    'TIP: You may pass just one value of repetitions')
            
            # Agent instanciation
            
            # Create a list of just one element in case there is only one, so
            # code in the loop can be simplified (epsilons[i], exp_types[i],
            # repetitions[i])
            if(type(epsilons) == float):
                temp = []
                for i in range(agent_num):
                    temp.append(epsilons)
                epsilons = temp
            if(type(exp_types) == str):
                temp = []
                for i in range(agent_num):
                    temp.append(exp_types)
                exp_types = temp
            if(type(repetitions) == int):
                temp = []
                for i in range(agent_num):
                    temp.append(repetitions)
                repetitions = temp
            
            for i in range(agent_num):
                """
                Create one list of turples, each turple contains data on an
                instance of an agent. The list contains agents with equal
                parameters.
                """
                agents.append(create_agents(
                    exp_types[i], epsilons[i], repetitions[i], gammas[i],
                    obs_size, n_actions, exploration_func, alg[a]))
        
        else: # Error handling for other cases
            raise KeyboardInterrupt(
                'Error: Unexpected parameter types or values')
        
        # Reassing unused parameters for next algorithm
        if(alg[a] in 'TRPO'):
            exp_types = exp_types_save
            epsilons = epsilons_save
            print('Warning: Not using exp_types and epsilons parameters')
    
    return agents

    