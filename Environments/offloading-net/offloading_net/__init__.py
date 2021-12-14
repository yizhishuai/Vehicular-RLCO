# -*- coding: utf-8 -*-
"""
Created on Sat Nov  13 16:39:00 2021

@author: Mieszko Ferens
"""

from gym.envs.registration import register

register(
    id='offload-v0',
    entry_point='offloading_net.envs:offload_netEnv',
)

