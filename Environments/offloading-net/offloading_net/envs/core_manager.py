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
        
        # Lists that hold info on reservations for all cores (except cloud)
        # Current remaining number of possible reservations
        self.queue_limit = []
        # Times at with each reservation will finish processing
        self.reserv_end = []
        
        # Limit for relative core load calculation (in ms)
        self.time_limit = 14
    
    def reserve(self, action, forward_delay, proc_delay, return_delay):
        
        # Look for time slots
        available = []
        next_core = []
        for core, f in enumerate(self.slots_duration[action]):
            # If core queue has too many reservations ignore it
            if(self.queue_limit[action][core] == 0):
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
        
        # Register the reservation
        self.queue_limit[action][core] -= 1
        
        # Assign the reservation to the first available time slot
        # If arrival of data is prior to slot's start
        if(forward_delay <= self.slots_start[action][core][slot]):
            # Shorten the slot
            self.slots_start[action][core][slot] += proc_delay
            self.slots_duration[action][core][slot] -= proc_delay
            # Register the reservation's end time
            self.reserv_end[action][core] = np.sort(np.append(
                self.reserv_end[action][core],
                self.slots_start[action][core][slot]))
            # Delete the slot if it is null (unless it is the end of the queue)
            if(self.slots_duration[action][core][slot] == 0 and
               len(self.slots_start[action][core]) - 1 != slot):
                self.slots_start[action][core] = np.delete(
                    self.slots_start[action][core], slot)
                self.slots_duration[action][core] = np.delete(
                    self.slots_duration[action][core], slot)
            # Negative values for slot duration should not be possible
            elif(self.slots_duration[action][core][slot] < 0):
                raise KeyboardInterrupt(
                    "Trouble with precision at core queues!")
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
            # Register the reservation's end time
            self.reserv_end[action][core] = np.sort(np.append(
                self.reserv_end[action][core],
                self.slots_start[action][core][slot+1]))
            # Delete the slot if it is null (unless it is the end of the queue)
            if(self.slots_duration[action][core][slot+1] == 0 and
               len(self.slots_start[action][core]) - 1 != slot+1):
                self.slots_start[action][core] = np.delete(
                    self.slots_start[action][core], slot+1)
                self.slots_duration[action][core] = np.delete(
                    self.slots_duration[action][core], slot+1)
            # Negative values for slot duration should not be possible
            elif(self.slots_duration[action][core][slot+1] < 0):
                raise KeyboardInterrupt(
                    "Trouble with precision at core queues!")
        
        return total_delay
    
    def update_and_calc_obs(self, app_time, precision_limit):
        
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
                self.reserv_end[node][core] -= app_time
                # Check if any reservation has ended
                ended = len(np.where(self.reserv_end[node][core] <= 0)[0])
                if(ended > 0):
                    self.queue_limit[node][core] += ended
                    self.reserv_end[node][core] = np.delete(
                        self.reserv_end[node][core], range(ended))
                # Check if any slot has ended
                if(np.amax(self.slots_start[node][core]) < 0):
                    loop = range(len(self.slots_start[node][core]))
                else:
                    loop = range(np.argmin(self.slots_start[node][core] < 0))
                for i in reversed(loop):
                    diff = np.around(self.slots_duration[node][core][i] +
                                     self.slots_start[node][core][i],
                                     precision_limit)
                    if(diff > 0): # If there is still time remaining
                        self.slots_start[node][core][i] = 0
                        self.slots_duration[node][core][i] = diff
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
        self.queue_limit = []
        self.reserv_end = []
        for a in range(n_nodes):
            if(node_cores[a] > 0): # Ignore cloud nodes
                duration = []
                start = []
                reservations = []
                limit = [2]*node_cores[a]
                for i in range(node_cores[a]):
                    duration.append(
                        np.array([self.time_limit], dtype=np.float64))
                    start.append(np.array([0], dtype=np.float64))
                    reservations.append(np.array([], dtype=np.float64))
                self.slots_duration.append(duration)
                self.slots_start.append(start)
                self.queue_limit.append(limit)
                self.reserv_end.append(reservations)
    
    