# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:06:32 2021

@author: Mieszko Ferens
"""

import numpy as np

### Class for core management

class core_manager():
    
    def __init__(self):
        
        # List that holds individual core info for all nodes (except cloud)
        self.cores = []
        
        # Lists that hold available time slots info for all cores (except cloud)
        # Available time slots's duration
        self.slots_duration = []
        # Available time slots's start time
        self.slots_start = []
        
        # Limit for relative core load calculation (in ms)
        self.time_limit = 100
        
        # Limit of queue
        self.queue_limit = 100
    
    def reserve(self, action, forward_delay, proc_delay, return_delay):
        
        # Look for time slots
        available = []
        next_core = []
        for core, f in enumerate(self.slots_duration[action]):
            # If core queue has too many reservations ignore it
            if(len(self.slots_start[action][core]) >= self.queue_limit):
                available.append(np.array([])) # Empty append to match indexes
                continue
            # Look for slots with sufficient durations (if fully usable)
            available.append(np.argwhere(f > proc_delay))
            # Check if no slots are available
            if(available[core].size == 0):
                continue
            # Register first available slot from core
            for i, g in enumerate(available[core]):
                if(forward_delay < self.slots_start[action][core][g[0]]):
                    next_core.append(
                        [core, available[core][i][0],
                         self.slots_start[action][core][available[core][i][0]]])
                    break
                # Check if slots that cannot be used fully are sufficient
                elif(f[g[0]] > forward_delay -
                   self.slots_start[action][core][g[0]] + proc_delay):
                    next_core.append(
                        [core, available[core][i][0],
                         self.slots_start[action][core][available[core][i][0]]])
                    break
        
        # Check if any slot is available
        if(len(next_core) == 0):
            return -1 # If processing was not possible return -1
        
        # Order the time slots by minimal delay
        next_core.sort(key=lambda x:x[2])
        core = next_core[0][0]
        slot = next_core[0][1]
        
        # Calculate the total delay
        total_delay = (max(forward_delay, self.slots_start[action][core][slot])
                       + proc_delay + return_delay)
        
        # Assign the reservation to the first available time slot
        # If arrival of data is prior to slot's start
        if(forward_delay <= self.slots_start[action][core][slot]):
            # Shorten the slot
            self.slots_start[action][core][slot] += proc_delay
            self.slots_duration[action][core][slot] -= proc_delay
            # Delete the slot if it is null (unless it is the end of the queue)
            if(self.slots_duration[action][core][slot] == 0 and
               len(self.slots_start[action][core]) - 1 != slot):
                self.slots_start[action][core] = np.delete(
                    self.slots_start[action][core], slot)
                self.slots_duration[action][core] = np.delete(
                    self.slots_duration[action][core], slot)
            # Negative values for slot duration should not be possible
            elif(self.slots_duration[action][core][slot] < 0):
                print("Trouble with precision at core queues!")
                raise KeyboardInterrupt
        # If arrival of data is posterior to slot's start
        else:
            # Create new slot before the new reservation
            self.slots_start[action][core] = np.insert(
                self.slots_start[action][core], slot,
                self.slots_start[action][core][slot])
            self.slots_duration[action][core] = np.insert(
                self.slots_duration[action][core], slot,
                forward_delay - self.slots_start[action][core][slot])
            # Shorten the old slot
            self.slots_start[action][core][slot+1] = (
                forward_delay + proc_delay)
            self.slots_duration[action][core][slot+1] -= (
                self.slots_duration[action][core][slot] + proc_delay)
            # Delete the slot if it is null (unless it is the end of the queue)
            if(self.slots_duration[action][core][slot+1] == 0 and
               len(self.slots_start[action][core]) - 1 != slot+1):
                self.slots_start[action][core] = np.delete(
                    self.slots_start[action][core], slot+1)
                self.slots_duration[action][core] = np.delete(
                    self.slots_duration[action][core], slot+1)
            # Negative values for slot duration should not be possible
            elif(self.slots_duration[action][core][slot+1] < 0):
                print("Trouble with precision at core queues!")
                raise KeyboardInterrupt
        
        return total_delay
    
    def update_and_calc_obs(self, app_time):
        
        ## Observation calculation
        obs = np.array([], dtype=np.float32)
        # Create an array with each element being a core (the self.cores
        # variable is an irregular array so a loop is necessary)
        for node in range(len(self.slots_start)):
            core_load = []
            for core in range(len(self.slots_start[node])):
                # As time passes the core's queue advances
                self.slots_start[node][core] -= app_time
                self.slots_duration[node][core][-1] += app_time
                # Check if any slot has ended
                for i in reversed(
                        range(np.argmax(self.slots_start[node][core] >= 0))):
                    diff = (self.slots_duration[node][core][i] +
                            self.slots_start[node][core][i])
                    if(diff > 0): # If there is still time remaining
                        self.slots_start[node][core][i] = 0
                        self.slots_duration[node][core][i] -= diff
                    else: # If the slot is over (all prior slots too)
                        self.slots_start[node][core] = np.delete(
                            self.slots_start[node][core], range(i+1))
                        self.slots_duration[node][core] = np.delete(
                            self.slots_duration[node][core], range(i+1))
                        break
                # Calculate load of the core
                core_load.append(1 - (
                    np.sum(self.slots_duration[node][core])/self.time_limit))
            # Add core load to observation
            obs = np.append(obs, np.array(core_load, dtype=np.float32))
        
        return obs
    
    def reset(self, n_nodes, node_cores):
        
        # Reset all cores
        self.slots_duration = []
        self.slots_start = []
        for a in range(n_nodes):
            if(node_cores[a] > 0): # Ignore cloud nodes
                duration = []
                start = []
                for i in range(node_cores[a]):
                    duration.append(
                        np.array([self.time_limit], dtype=np.float64))
                    start.append(np.array([0], dtype=np.float64))
                self.slots_duration.append(duration)
                self.slots_start.append(start)
    
    