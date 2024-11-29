# a decorator that takes in a function and returns a new function

import threading
import time
import torch
import power_profile
import nvpmplus
import numpy as np
import warnings


# global variables
BATCH_SIZE = 16 # default batch size. used by the service thread. can be changed by the optimizer

global logs_text_prefix
logs_text_prefix = "/dev/null/" # set to /dev/null/ to disable logging

# uncomment the following line to enable logging
# logs_text_prefix = "results/" + time.strftime("%Y%m%d-%H%M%S") + "_" + logs_text_prefix

class EnergyOptimizer_skeleton:
	"""
	An abstract class that defines the skeleton of an energy optimizer. This is the base class for all the energy optimizers. It has the basic functions that are common to all the optimizers.
	"""
	def __init__(self, alpha=0.5, cache_length=416, arr_rate_thres_pcent=None):
		"""
		This is the constructor of the class. It initializes the class with the following parameters:
		alpha: weight of energy in the cost function. 1-alpha is the weight of time
		cache_length: number of inferences to cache before updating the history
		arr_rate_thres_pcent: if set, the optimizer will detect significant changes in arrival rate and call significant_change function
		"""
		self.alpha = alpha
		self.cache_length = cache_length
		self.cpu_values = list(range(len(nvpmplus.cpu_scaling_available_frequencies)))
		self.gpu_values = list(range(len(nvpmplus.gpu_available_frequencies)))
		self.batchsize_values = list(range(17))
		self.optim_T = 0
		self.cache_energy = []
		self.cache_time = []
		self.cpu_min_limit = 4
		self.gpu_min_limit = 4
		self.batchsize_min_limit = 4
		self.arr_rate_thres_pcent = arr_rate_thres_pcent
		
		self.logs_optim_fp = None
		self.energy_baseline = None
		self.time_baseline = None
		self.last_arrival_rate = None

		self.model_type = "resnet" # used by the queue service to batch the input values
		self.bert_tokenizer = None # will be used by the queue service

		# history is a matrix of shape (len(cpu_values), len(gpu_values), len(batchsize_values)). it stores "cost" of each configuration
		# np.nan means the configuration has not been evaluated yet
		self.history = [[[np.nan for _ in self.batchsize_values] for _ in self.gpu_values] for _ in self.cpu_values]
		self.history_optim_T = [[[np.nan for _ in self.batchsize_values] for _ in self.gpu_values] for _ in self.cpu_values]
		self.text_dimention_mapping = {"CPU":0, "GPU":1, "BATCHSIZE":2}

		self.starting_config = [self.cpu_values[-1], self.gpu_values[-1], self.batchsize_values[-1]]
		self.set_config(*self.starting_config, comment="Starting config")
	
	# can be used to set the baseline energy and time (optional)
	def set_baseline(self,IAT=0.050,fname = "master_reference_all_max.csv"): 
		"""
		Use this function to set the baseline energy and time. This is used to calculate the cost of the configurations.
		This is optional. If not set, the optimizer will use the third run of the max,max,max configuration as the baseline.
		IAT: inter-arrival time in seconds
		fname: file name of the master reference file
		"""

		# format of lines: IATs, init_energy, total_energy
		# IAT is the inter-arrival time in ms
		lines = open(fname).readlines()
		for i,line in enumerate(lines):
			if i==0:
				continue
			row = line.split(",")
			if float(row[0]) == IAT:
				self.energy_baseline = float(row[1])
				self.time_baseline = float(row[2])
				break
		print("baseline set",self.energy_baseline,self.time_baseline)

	def set_config(self, cpu, gpu, batchsize, comment=""):
		"""
		This function is used to set the configuration of the system. It sets the CPU and GPU frequencies and the batch size.
		cpu: index of the CPU frequency in the list of available CPU frequencies
		gpu: index of the GPU frequency in the list of available GPU frequencies
		batchsize: batch size
		comment: a comment to be added to the logs
		"""
		cpu, gpu, batchsize = int(cpu), int(gpu), int(batchsize)
		self.last_set_config = (cpu, gpu, batchsize)
		self.last_set_config_comment = comment
		# check if config is valid
		if cpu not in self.cpu_values or gpu not in self.gpu_values or batchsize not in self.batchsize_values or (not batchsize>0):
			raise ValueError("Invalid configuration. Tried config is: ", cpu, gpu, batchsize)

		nvpmplus.set_state(nvpmplus.cpu_lim, cpu, gpu)
		global BATCH_SIZE
		BATCH_SIZE = batchsize

	def set_governor(self, cpu_governor_index, gpu_governor_index, batchsize, comment=""):
		"""
		This function is used to set the governor of the system instead of the frequencies. Alternate to set_config.
		It sets the CPU and GPU governors and the batch size. Called by the service thread.
		cpu_governor_index: index of the CPU governor in the list of available CPU governors
		gpu_governor_index: index of the GPU governor in the list of available GPU governors
		batchsize: batch size
		comment: a comment to be added to the logs
		"""

		# govs are integers. they are the index of the governor in the list of available governors
		cpu_govs = nvpmplus.cpu_govs
		gpu_govs = nvpmplus.gpu_govs

		cpu = cpu_govs[cpu_governor_index]
		gpu = gpu_govs[gpu_governor_index]
		batchsize = int(batchsize)
		
		# config should always be in integers because it is used as an index in the history matrix
		self.last_set_config = (cpu_governor_index, gpu_governor_index, batchsize) 
		self.last_set_config_comment = comment

		# check if config is valid
		if cpu_governor_index not in range(len(cpu_govs)) or gpu_governor_index not in range(len(gpu_govs)) or batchsize not in self.batchsize_values or (not batchsize>0):
			raise ValueError("Invalid configuration. Tried config is: ", cpu_governor_index, gpu_governor_index, batchsize)
		
		nvpmplus.set_gov(cpu, gpu)
		global BATCH_SIZE
		BATCH_SIZE = batchsize

	def post_results(self, energy, time):
		"""
		This function is used to post the results of the last configuration. It caches the results for cache_length inferences and then updates the history.
		energy: energy consumed by the last configuration
		time: time taken by the last configuration
		"""
		self.cache_energy += energy
		self.cache_time += time
		if len(self.cache_energy) >= self.cache_length:

			if self.optim_T == 2 and self.energy_baseline==None: #setting baseline in the 3rd config if not already set
				self.energy_baseline = np.median(self.cache_energy[100:])
				self.time_baseline = np.median(self.cache_time[100:])

			self.update_history()
			
			if self.optim_T > 3: #dont run optimizer in the first 3 configurations
				self.run_optimizer()

			self.optim_T += 1

	def update_history(self):
		"""
		This function is used to update the history data structure. It calculates the cost of the last configuration and updates the history matrix.
		"""

		# calculate the cost of the last configuration
		mean_energy = np.median(self.cache_energy[100:])
		mean_time = np.median(self.cache_time[100:])
		
		try:
			cost = self.alpha*mean_energy/self.energy_baseline + (1-self.alpha)*mean_time/self.time_baseline
		except: # we might get here if baseline is not set yet
			cost = float("inf")

		print("last set config",self.last_set_config,"cost",cost, "last set config comment", self.last_set_config_comment)
		self.history[self.last_set_config[0]][self.last_set_config[1]][self.last_set_config[2]] = cost
		self.history_optim_T[self.last_set_config[0]][self.last_set_config[1]][self.last_set_config[2]] = self.optim_T
		self.save_logs_optim(mean_energy, mean_time, cost)
		self.cache_energy = []
		self.cache_time = []
	
	def arrival_rate_observer(self, batch_arr_ts):
		"""
		This function is used to observe the arrival rate of the requests. It is called by the service thread after every batch of requests. If the arrival rate changes significantly, it calls the significant_change_detected function.
		batch_arr_ts: list of timestamps of the requests in the batch
		"""
		if not self.arr_rate_thres_pcent:
			return
		
		total_time = batch_arr_ts[-1] - batch_arr_ts[0]
		total_arrivals = len(batch_arr_ts)
		arrival_rate = total_arrivals / total_time

		if self.last_arrival_rate is None:
			self.last_arrival_rate = arrival_rate
			return
		
		if np.abs(arrival_rate - self.last_arrival_rate)/self.last_arrival_rate > self.arr_rate_thres_pcent/100:
			print("significant change in arrival rate detected")
			self.significant_change_detected()
			self.last_arrival_rate = arrival_rate	

	def significant_change_detected(self):
		"""
		This function is called when a significant change in the arrival rate is detected. It is used to clear the history data structure. To be implemented by the inheriting class. Used for cleaning history data structure and states.
		"""
		pass

	def save_logs_optim(self, mean_energy, mean_time, cost):
		"""
		This function is used to save the logs of the optimization. It saves the energy, time, cost, and the queue size in a csv file.
		mean_energy: energy consumed by the last configuration
		mean_time: time taken by the last configuration
		cost: cost of the last configuration
		"""
		global logs_text_prefix, request_queue
		if self.logs_optim_fp is None:
			self.logs_optim_fp = open(logs_text_prefix+"logs_optim.csv", "w")
			self.logs_optim_fp.write("optim_T,cpu,gpu,batchsize,energy,time,cost,queue_size,comment\n")
		self.logs_optim_fp.write(f"{self.optim_T},{self.last_set_config[0]},{self.last_set_config[1]},{self.last_set_config[2]},{mean_energy},{mean_time},{cost},{len(request_queue)},{self.last_set_config_comment}\n")

	def optimizer_stop(self):
		"""
		This function is used to stop the optimizer. It closes the logs file.
		"""
		if self.logs_optim_fp is not None:
			self.logs_optim_fp.close()

	def run_optimizer(self):
		"""
		This function is used to run the optimizer. It is called after every cache_length inferences. To be implemented by the inheriting class.
		"""
		# to be made by inheriting class
		pass

class EnergyOptimizer_random(EnergyOptimizer_skeleton):
	"""
	This class is used to implement a random energy optimizer. It chooses a random configuration at every step.
	"""
	def __init__(self, cache_length=416):
		"""
		This is the constructor of the class. It initializes the class with the following parameters:
		cache_length: number of inferences to cache before updating the history
		"""
		super().__init__(cache_length=cache_length)

	# only for testing purposes
	def run_optimizer(self):
		"""
		This function is used to run the optimizer. It is called after every cache_length inferences. It chooses a random configuration at every step.
		"""

		# get a random configuration
		cpu = np.random.choice(self.cpu_values)
		gpu = np.random.choice(self.gpu_values)
		batchsize = np.random.choice([x for x in self.batchsize_values if x > 0])
		print("random config",cpu, gpu, batchsize)
		self.set_config(cpu, gpu, batchsize, comment="random")

class EnergyOptimizer_fixed(EnergyOptimizer_skeleton):
	"""
	This class is used to implement a fixed energy optimizer. It chooses a fixed pre-defined configuration at every step.
	"""
	def __init__(self, cpu, gpu, batchsize, cache_length=416):
		"""
		This is the constructor of the class. It initializes the class with the following parameters:
		cpu: index of the CPU frequency in the list of available CPU frequencies
		gpu: index of the GPU frequency in the list of available GPU frequencies
		batchsize: batch size
		cache_length: number of inferences to cache before updating the history
		"""
		self.optimizer_queue = []
		super().__init__(cache_length=cache_length)
		self.set_config(cpu, gpu, batchsize, comment="Fixed config")

	def run_optimizer(self):
		"""
		Does nothing. Leaves the configuration as it is.
		"""
		pass

class EnergyOptimizer_DVFS(EnergyOptimizer_skeleton):
	"""
	This class is used to implement a DVFS energy optimizer. It uses preinstalled DVFS governors to set the CPU and GPU frequencies.
	"""
	def __init__(self, cpu_governor_index=6, gpu_governor_index=1, batchsize=16, cache_length=416):
		"""
		This is the constructor of the class. It initializes the class with the following parameters:
		cpu_governor_index: index of the CPU governor in the list of available CPU governors
		gpu_governor_index: index of the GPU governor in the list of available GPU governors
		batchsize: batch size
		cache_length: number of inferences to cache before updating the history
		"""
		self.optimizer_queue = []
		super().__init__(cache_length=cache_length)
		self.set_governor(cpu_governor_index, gpu_governor_index, batchsize, comment="Fixed config")

	def run_optimizer(self):
		# only set the fixed configuration
		pass

class EnergyOptimizer_linearsweeps(EnergyOptimizer_skeleton):
	"""
	This class is used to implement a linear sweeps energy optimizer. It sweeps through the CPU, GPU, and batchsize values in a linear fashion one after the other. After each sweep, it chooses the best configuration for that parameter and sets it.
	"""
	def __init__(self, last_sweep="BATCHSIZE", sweep_next_mapping={"BATCHSIZE":"CPU", "CPU":"GPU", "GPU":"BATCHSIZE"},cache_length=416):
		"""
		This is the constructor of the class. It initializes the class with the following parameters:
		last_sweep: the last parameter that was swept. It can be "CPU", "GPU", or "BATCHSIZE"
		sweep_next_mapping: a dictionary that maps the last parameter to the next parameter to sweep
		cache_length: number of inferences to cache before updating the history
		"""
		self.last_sweep = last_sweep
		self.sweep_next_mapping = sweep_next_mapping
		self.optimizer_queue = []
		self.num_sweeps_done = 0		

		super().__init__(cache_length=cache_length)
		self.last_sweep_backup = [self.starting_config] # used to backup all the configurations of the last sweep
		
	def run_optimizer(self):
		"""
		This function is used to run the optimizer. It is called after every cache_length inferences. It sweeps through the CPU, GPU, and batchsize values in a linear fashion one after the other. After each sweep, it chooses the best configuration for that parameter and sets it.
		"""
		if len(self.optimizer_queue) > 0:
			self.set_config(*self.optimizer_queue.pop(0), comment="queue pop")
			return

		next_sweep = self.sweep_next_mapping[self.last_sweep]

		if self.num_sweeps_done < 3:
			self.num_sweeps_done += 1

			if next_sweep == "CPU":
				best_cost = float("inf")
				best_config = (0,0,0)
				for i in self.last_sweep_backup:
					if self.history[i[0]][i[1]][i[2]] < best_cost:
						best_cost = self.history[i[0]][i[1]][i[2]]
						best_config = i

				self.optimizer_queue = [(x, best_config[1], best_config[2]) for x in self.cpu_values if x>=self.cpu_min_limit]
				self.last_sweep_backup = [(x, best_config[1], best_config[2]) for x in self.cpu_values if x>=self.cpu_min_limit]
				self.set_config(*self.optimizer_queue.pop(0), comment="CPU sweep")
				self.last_sweep = "CPU"
				return
			
			if next_sweep == "GPU":
				best_cost = float("inf")
				best_config = (0,0,0)
				for i in self.last_sweep_backup:
					if self.history[i[0]][i[1]][i[2]] < best_cost:
						best_cost = self.history[i[0]][i[1]][i[2]]
						best_config = i

				self.optimizer_queue = [(best_config[0], y, best_config[2]) for y in self.gpu_values if y>=self.gpu_min_limit]
				self.last_sweep_backup = [(best_config[0], y, best_config[2]) for y in self.gpu_values if y>=self.gpu_min_limit]
				self.set_config(*self.optimizer_queue.pop(0), comment="GPU sweep")
				self.last_sweep = "GPU"
				return
			
			if next_sweep == "BATCHSIZE":				
				best_cost = float("inf")
				best_config = (0,0,0)
				for i in self.last_sweep_backup:
					if self.history[i[0]][i[1]][i[2]] < best_cost:
						best_cost = self.history[i[0]][i[1]][i[2]]
						best_config = i

				self.optimizer_queue = [(best_config[0], best_config[1], z) for z in self.batchsize_values if z>=self.batchsize_min_limit]
				self.last_sweep_backup = [(best_config[0], best_config[1], z) for z in self.batchsize_values if z>=self.batchsize_min_limit]
				self.set_config(*self.optimizer_queue.pop(0), comment="BATCHSIZE sweep")
				self.last_sweep = "BATCHSIZE"
				return
			
		else:
			# if all sweeps are done, we pick the best configuration and use it forever
			best_config = (0,0,0)
			best_cost = float("inf")
			for i in self.cpu_values:
				for j in self.gpu_values:
					for k in self.batchsize_values:
						if self.history[i][j][k] < best_cost:
							best_cost = self.history[i][j][k]
							best_config = (i,j,k)
			self.optimizer_queue = [best_config]*20
			self.set_config(*self.optimizer_queue.pop(0), comment="best config")
			return

class EnergyOptimizer_GridSearch(EnergyOptimizer_skeleton):
	"""
	This class is used to implement a grid search energy optimizer. It tries all (but alternative) possible configurations in a grid search fashion. It chooses the best configuration and uses it forever. This is not a practical optimizer because it doesn't adapt to the changing environment.
	"""
	def __init__(self, cache_length=416):
		"""
		This is the constructor of the class. It initializes the class with the following parameters:
		cache_length: number of inferences to cache before updating the history
		"""
		self.optimizer_queue = []
		self.num_sweeps_done = 0
		self.grid_search_done = False
		super().__init__(cache_length=cache_length)
		
	def run_optimizer(self):
		"""
		This function is used to run the optimizer. It is called after every cache_length inferences. It tries all (but alternative) possible configurations in a grid search fashion. It chooses the best configuration and uses it forever.
		"""
		if len(self.optimizer_queue) > 0:
			self.set_config(*self.optimizer_queue.pop(0), comment="queue pop")
			
			# empty the queue everytime we change the configuration
			global request_queue, request_queue_ts
			request_queue = []
			request_queue_ts = []

			return
		
		if self.grid_search_done:
			# kill the entire python process but flush the logs first
			self.optimizer_stop()
			global logs_tasks_fp
			if logs_tasks_fp is not None:
				logs_tasks_fp.close()

			import os
			os._exit(0)

		if self.num_sweeps_done < 1:
			self.num_sweeps_done += 1
			self.optimizer_queue = [(x, y, z) for x in self.cpu_values if (x%2==0 and x!=min(self.cpu_values))
						   				for y in self.gpu_values if (y%2==0 and y!=min(self.gpu_values))
										for z in self.batchsize_values if (z%2==0 and z!=min(self.batchsize_values)) and z > 0]
			self.set_config(*self.optimizer_queue.pop(0), comment="gridsearch")

		else:
			# pick the best configuration and use it forever
			best_config = (0,0,0)
			best_cost = float("inf")
			for i in self.cpu_values:
				for j in self.gpu_values:
					for k in self.batchsize_values:
						if self.history[i][j][k] < best_cost:
							best_cost = self.history[i][j][k]
							best_config = (i,j,k)
			self.optimizer_queue = [best_config]*20
			self.set_config(*self.optimizer_queue.pop(0), comment="best config")
			self.grid_search_done = True
			return

class EnergyOptimizer_MAB_multiDim(EnergyOptimizer_skeleton):
	"""
	This class is used to implement a multi-armed bandit energy optimizer. It uses a multi-armed bandit algorithm to choose the next configuration. It chooses the best configuration for each parameter independently and sets it in a round-robin fashion. It keeps a history of costs for each parameter separately.
	"""
	def __init__(self, order_dimention=["CPU", "GPU", "BATCHSIZE"], exp_avg_alpha=0.9, hot_start=10, exploit_prob=0.9,cache_length=416):
		"""
		This is the constructor of the class. It initializes the class with the following parameters:
		order_dimention: order of the dimention to be optimized. It can be ["CPU", "GPU", "BATCHSIZE"] or any permutation of it.
		exp_avg_alpha: exponential average alpha
		exploit_prob: probability of exploiting the best configuration for a dimention
		hot_start: if set to boolean True, does a full gridsearch. If set to an integer, does a random search for that many configurations
		cache_length: number of inferences to cache before updating the history
		"""
		self.order_dimention = order_dimention 	# order of the dimention to be optimized
		self.exp_avg_alpha = exp_avg_alpha 		# exponential average alpha
		self.exploit_prob = exploit_prob 		# probability of exploiting the best configuration for a dimention
		self.hot_start = hot_start 		# if set to boolean True, does a full gridsearch. If set to an integer, does a random search for that many configurations

		super().__init__(cache_length=cache_length)
		self.optimizer_queue = []
		self.order_offset = None
	
	def update_history(self):
		"""
		This function is used to update the history data structure. It calculates the cost of the last configuration and updates the history matrix. This is different from the base class because it uses an exponential average to update the cost instead of just overwriting it.
		"""
		mean_energy = np.median(self.cache_energy[100:])
		mean_time = np.median(self.cache_time[100:])
		
		try:
			cost = self.alpha*mean_energy/self.energy_baseline + (1-self.alpha)*mean_time/self.time_baseline
		except: # we might get here if baseline is not set yet
			cost = float("inf")

		print("last set config",self.last_set_config,"cost",cost, "last set config comment", self.last_set_config_comment)
		if np.isnan(self.history[self.last_set_config[0]][self.last_set_config[1]][self.last_set_config[2]]):
			self.history[self.last_set_config[0]][self.last_set_config[1]][self.last_set_config[2]] = cost
		else:
			self.history[self.last_set_config[0]][self.last_set_config[1]][self.last_set_config[2]] = self.history[self.last_set_config[0]][self.last_set_config[1]][self.last_set_config[2]] * (1 - self.exp_avg_alpha) + cost * self.exp_avg_alpha

		print("new history",self.history[self.last_set_config[0]][self.last_set_config[1]][self.last_set_config[2]])

		self.history_optim_T[self.last_set_config[0]][self.last_set_config[1]][self.last_set_config[2]] = self.optim_T
		self.save_logs_optim(mean_energy, mean_time, cost)
		self.cache_energy = []
		self.cache_time = []

	def run_optimizer(self):
		"""
		This function is used to run the optimizer. It is called after every cache_length inferences. It chooses the best configuration for each parameter independently and sets it in a round-robin fashion. It keeps a history of costs for each parameter separately.
		"""
		if len(self.optimizer_queue) > 0:
			self.set_config(*self.optimizer_queue.pop(0), comment="queue pop")
			return
		
		# if hot_start is set to boolean True, do a full gridsearch
		if self.hot_start == True and type(self.hot_start) == bool:
			self.optimizer_queue = [(x, y, z) for x in self.cpu_values for y in self.gpu_values for z in self.batchsize_values 
						   				if (x>=self.cpu_min_limit and y>=self.gpu_min_limit and z>=self.batchsize_min_limit)]
			self.hot_start = 0
			self.set_config(*self.optimizer_queue.pop(0), comment="gridsearch")
			return
		
		# if hot_start is set to an integer, do a random search for that many configurations
		if self.hot_start > 0:
			self.optimizer_queue = [(	np.random.choice([x for x in self.cpu_values if x >= self.cpu_min_limit]), 
										np.random.choice([y for y in self.gpu_values if y >= self.gpu_min_limit]), 
										np.random.choice([z for z in self.batchsize_values if z >= self.batchsize_min_limit]))
													for _ in range(self.hot_start)]
			self.hot_start = 0
			self.set_config(*self.optimizer_queue.pop(0), comment="random search")
			return
		
		# choose the best configuration for the dimention and set it. other dimention configurations are set to the last set configuration
		if self.order_offset is None:
			self.order_offset = self.optim_T % 3
		dimention_to_set = self.order_dimention[(self.optim_T - self.order_offset) % 3]

		if dimention_to_set == "CPU":
			best_cost = float("inf")
			best_config = (0,0,0)

			if np.random.rand() < self.exploit_prob:
				# get the best configuration for CPU. mean over all other dimention configurations
				for i in self.cpu_values:
					with warnings.catch_warnings():
						warnings.simplefilter("ignore", category=RuntimeWarning)
						cost = np.nanmean([self.history[i][j][k] for j in self.gpu_values for k in self.batchsize_values if k > 0])
					if cost < best_cost:
						best_cost = cost
						best_config = (i, self.last_set_config[1], self.last_set_config[2])
				comment = "CPU exploit"
			else:
				best_config = (np.random.choice([x for x in self.cpu_values if x >= self.cpu_min_limit]), self.last_set_config[1], self.last_set_config[2])
				comment = "CPU random"

			self.set_config(*best_config, comment=comment)
			return
		
		elif dimention_to_set == "GPU":
			best_cost = float("inf")
			best_config = (0,0,0)

			if np.random.rand() < self.exploit_prob:
				# get the best configuration for GPU. mean over all other dimention configurations
				for j in self.gpu_values:
					with warnings.catch_warnings():
						warnings.simplefilter("ignore", category=RuntimeWarning)
						cost = np.nanmean([self.history[i][j][k] for i in self.cpu_values for k in self.batchsize_values if k > 0])
					if cost < best_cost:
						best_cost = cost
						best_config = (self.last_set_config[0], j, self.last_set_config[2])
				comment = "GPU exploit"
			else:
				best_config = (self.last_set_config[0], np.random.choice([y for y in self.gpu_values if y >= self.gpu_min_limit]), self.last_set_config[2])
				comment = "GPU random"

			self.set_config(*best_config, comment=comment)
			return
		
		elif dimention_to_set == "BATCHSIZE":
			best_cost = float("inf")
			best_config = (0,0,0)

			if np.random.rand() < self.exploit_prob:
				# get the best configuration for BATCHSIZE. mean over all other dimention configurations
				for k in self.batchsize_values[1:]:
					with warnings.catch_warnings():
						warnings.simplefilter("ignore", category=RuntimeWarning)
						cost = np.nanmean([self.history[i][j][k] for i in self.cpu_values for j in self.gpu_values if k > 0])
					if cost < best_cost:
						best_cost = cost
						best_config = (self.last_set_config[0], self.last_set_config[1], k)
				comment = "BATCHSIZE exploit"
			else:
				best_config = (self.last_set_config[0], self.last_set_config[1], np.random.choice([z for z in self.batchsize_values if z >= self.batchsize_min_limit]))
				comment = "BATCHSIZE random"

			self.set_config(*best_config, comment=comment)
			return

class EnergyOptimizer_MAB_multiDim_all_at_once(EnergyOptimizer_MAB_multiDim):
	"""
	This class is used to implement a multi armed bandit 'all at once' energy optimizer. It uses a multi-armed bandit algorithm to choose the next configuration. It chooses the best configuration for all dimention configurations at once. It keeps a history of costs for each configuration. Same as MAB_multiDim, but instead of setting one dimention at a time, it sets all dimention configurations at once.
	"""
	def run_optimizer(self):
		"""
		This function is used to run the optimizer. It is called after every cache_length inferences. It chooses the best configuration for all dimention configurations at once. It keeps a history of costs for each configuration.
		"""
		if len(self.optimizer_queue) > 0:
			self.set_config(*self.optimizer_queue.pop(0), comment="queue pop")
			return
		
		# if hot_start is set to boolean True, do a full gridsearch
		if self.hot_start == True and type(self.hot_start) == bool:
			self.optimizer_queue = [(	np.random.choice([x for x in self.cpu_values if x >= self.cpu_min_limit]), 
										np.random.choice([y for y in self.gpu_values if y >= self.gpu_min_limit]), 
										np.random.choice([z for z in self.batchsize_values if z >= self.batchsize_min_limit]))
													for _ in range(self.hot_start)]
			self.hot_start = 0
			self.set_config(*self.optimizer_queue.pop(0), comment="gridsearch")
			return
		
		# if hot_start is set to an integer, do a random search for that many configurations
		if self.hot_start > 0:
			self.optimizer_queue = [(	np.random.choice([x for x in self.cpu_values if x >= self.cpu_min_limit]), 
										np.random.choice([y for y in self.gpu_values if y >= self.gpu_min_limit]), 
										np.random.choice([z for z in self.batchsize_values if z >= self.batchsize_min_limit]))
													for _ in range(self.hot_start)]
			self.hot_start = 0
			self.set_config(*self.optimizer_queue.pop(0), comment="random search")
			return
		
		if np.random.rand() < self.exploit_prob:
			# get the best configuration for all dimention. mean over all other dimention configurations
			best_cost = float("inf")
			best_config = (0,0,0)
			for i in self.cpu_values:
				for j in self.gpu_values:
					for k in self.batchsize_values:
						if self.history[i][j][k] < best_cost:
							best_cost = self.history[i][j][k]
							best_config = (i,j,k)
			comment = "all dimention exploit"
			if best_config == (0,0,0):
				print(self.history)
				print("something went wrong in all dimention exploit")
		else:
			best_config = (	np.random.choice([x for x in self.cpu_values if x >= self.cpu_min_limit]), 
							np.random.choice([y for y in self.gpu_values if y >= self.gpu_min_limit]), 
							np.random.choice([z for z in self.batchsize_values if z >= self.batchsize_min_limit]))
			comment = "all dimension random"

		self.set_config(*best_config, comment=comment)
		return

class EnergyOptimizer_Gradient_Descent(EnergyOptimizer_skeleton):
	"""
	This class is used to implement a gradient descent energy optimizer. It does gradient descent on the cost function. At every config, it explores (measures) the neighbourhood and chooses the best configuration. the corners in CPU-GPU plane are calculated but not measured. This is the 'EcoGD' optimizer mentioned in the paper.
	"""
	def __init__(self,memory_limit=20, max_loops=10, cache_length=416, jump_learn_factor=None, arr_rate_thres_pcent=None):
		"""
		This is the constructor of the class. It initializes the class with the following parameters:
		memory_limit: number of previous configurations to remember
		max_loops: max number of loops to run the optimizer if it's stuck
		cache_length: number of inferences to cache before updating the history
		jump_learn_factor: factor to jump learn the optimizer
		arr_rate_thres_pcent: threshold for significant change in arrival rate
		"""
		self.optimizer_queue = []
		self.memory_limit = memory_limit
		self.max_loops = max_loops
		self.jump_learn_factor = jump_learn_factor
		self.last_center = None
		self.loop_counter = 0
		super().__init__(cache_length=cache_length, arr_rate_thres_pcent=arr_rate_thres_pcent)
	
	def significant_change_detected(self):
		"""
		This function is called when a significant change in the arrival rate is detected. It is used to clear the history data structure. Used for cleaning history data structure and states
		"""
		print("significant change detected. clearing history")
		self.history = [[[np.nan for _ in self.batchsize_values] for _ in self.gpu_values] for _ in self.cpu_values]
		self.history_optim_T = [[[np.nan for _ in self.batchsize_values] for _ in self.gpu_values] for _ in self.cpu_values]

	def run_optimizer(self):
		"""
		This function is used to run the optimizer. It is called after every cache_length inferences. It does gradient descent on the cost function. At every config, it explores (measures) the neighbourhood and chooses the best configuration. the corners in CPU-GPU plane are calculated but not measured. This is the 'EcoGD' optimizer mentioned in the paper
		"""
		if len(self.optimizer_queue) > 0:
			self.set_config(*self.optimizer_queue.pop(0), comment="queue pop")
			return
		
		if self.last_center is None:
			self.last_center = self.last_set_config # starting config. TODO: randomize this

		# see if we have already explored the neighbourhood of the last center. but not the corners
		valid_side_neighbours = \
				[(self.last_center[0]+i, self.last_center[1], self.last_center[2]) for i in [-1,1] if self.last_center[0]+i in self.cpu_values] \
				+ [(self.last_center[0], self.last_center[1]+j, self.last_center[2]) for j in [-1,1] if self.last_center[1]+j in self.gpu_values] \
				+ [(self.last_center[0], self.last_center[1], self.last_center[2]+k) for k in [-1,1] if self.last_center[2]+k in self.batchsize_values 
	   																											and self.last_center[2]+k > 0]
		check_side_neighbours = [((self.history[i][j][k]!=np.nan) and (self.optim_T - self.history_optim_T[i][j][k] < self.memory_limit))
					  				for i,j,k in valid_side_neighbours]

		if not all(check_side_neighbours):
			# if not, measure the neighbourhood that is not measured yet
			self.optimizer_queue = [valid_side_neighbours[i] for i in range(len(valid_side_neighbours)) if not check_side_neighbours[i]]
			self.set_config(*self.optimizer_queue.pop(0), comment="explore neighbourhood")
			return
		
		# estimate cost of the corners using the cost of the side neighbours. 
		# create a matrix of shape (3,3,3) where each element is the cost of the corner or side neighbour. np.inf if corner does not exist
		pseudo_neighbour_history = np.inf * np.ones((3,3,3))
		for n in valid_side_neighbours:
			i,j,k = n
			pseudo_neighbour_history[i-self.last_center[0]+1][j-self.last_center[1]+1][k-self.last_center[2]+1] = self.history[i][j][k]
		
		# calculate the corners. but only among freqs not batchsize
		valid_corners = [(self.last_center[0]+i, self.last_center[1]+j, self.last_center[2]) for i in [-1,1] for j in [-1,1]
								if (self.last_center[0]+i in self.cpu_values) and (self.last_center[1]+j in self.gpu_values)]
		
		# calculate the cost of the corners
		corner_costs = []
		for i,j,k in valid_corners:
			# get the side neighbours of the corner
			corner_side_neighbours = [(i+x,j+y,k) for x in [-1,1] for y in [-1,1] if (i+x,j+y,k) in valid_side_neighbours]
			# calculate the cost of the corner. delta from center is sum of the deltas costs of the side neighbours
			corner_costs.append(np.mean([pseudo_neighbour_history[x-i+1][y-j+1][k] for x,y,k in corner_side_neighbours]))
			# corner_costs.append(np.mean([self.history[x][y][k] for x,y,k in corner_side_neighbours]))
		
		for n in valid_corners:
			i,j,k = n
			pseudo_neighbour_history[i-self.last_center[0]+1][j-self.last_center[1]+1][k-self.last_center[2]+1] = corner_costs.pop(0)
		
		# add self to the pseudo_neighbour_history
		pseudo_neighbour_history[1][1][1] = self.history[self.last_center[0]][self.last_center[1]][self.last_center[2]]


		# get the best neighbour (both side and corner) and set it
		best_cost = np.inf
		best_config = (1,1,1)
		for i in range(3):
			for j in range(3):
				for k in range(3):
					if pseudo_neighbour_history[i][j][k] < best_cost:
						best_cost = pseudo_neighbour_history[i][j][k]
						best_config = (self.last_center[0]+i-1, self.last_center[1]+j-1, self.last_center[2]+k-1)
		
		# in cpu, gpu plane, make a based on jump_learn_factor
		if self.jump_learn_factor:
			# calculate the jump size using self.jump_learn_factor and the change in cost of the best neighbour
			change_in_cost = (pseudo_neighbour_history[1][1][1] - best_cost)/pseudo_neighbour_history[1][1][1]
			self.jump_size = int(np.ceil(self.jump_learn_factor*change_in_cost))
			self.jump_size = np.clip(self.jump_size, 1, 5)
			self.jump_size = (self.jump_size, self.jump_size)
			print("jump_learn_factor", self.jump_learn_factor, "change_in_cost", change_in_cost, "jump_size", self.jump_size)
			
		else:
			self.jump_size = (1,1)

		diff_config = [best_config[0]-self.last_center[0], best_config[1]-self.last_center[1], best_config[2]-self.last_center[2]]
		diff_config[0] = diff_config[0] * self.jump_size[0]
		diff_config[1] = diff_config[1] * self.jump_size[1]
		best_config = (self.last_center[0]+diff_config[0], self.last_center[1]+diff_config[1], self.last_center[2]+diff_config[2])

		# clip the config to the limits
		best_config = (np.clip(best_config[0], self.cpu_min_limit, self.cpu_values[-1]), 
						np.clip(best_config[1], self.gpu_min_limit, self.gpu_values[-1]),
						np.clip(best_config[2], self.batchsize_min_limit, self.batchsize_values[-1]))

		if best_config == self.last_center:
			# if the best config is the same as the last center, we are done
			self.loop_counter += 1
			if self.loop_counter >= self.max_loops:
				# if we have done the max number of loops, random neighbour
				self.loop_counter = 0
				all_neighbours = valid_corners + valid_side_neighbours
				best_config = all_neighbours[np.random.choice(len(all_neighbours))]
				self.last_center = best_config
				self.set_config(*best_config, comment="random neighbour; max_loops reached")
				return
		else:
			self.loop_counter = 0

		self.last_center = best_config
		jump_size_str = str(self.jump_size[0]) + ";" + str(self.jump_size[1])
		self.set_config(*best_config, comment="best in neighbourhood with jump size "+jump_size_str)
		return


class EnergyOptimizer_BayesianOptimization(EnergyOptimizer_skeleton):
	"""
	This class is used to implement a bayesian optimization energy optimizer. It uses bayesian optimization to find the best configuration. It uses the GaussianProcessRegressor from sklearn to model the cost function and the expected improvement to find the best configuration.
	"""
	from sklearn.gaussian_process import GaussianProcessRegressor
	from sklearn.gaussian_process.kernels import RBF
	from scipy.stats import norm

	def expected_improvement(self, x, gp_model, best_y):
		"""
		This function is used to calculate the expected improvement of a configuration. It uses the GaussianProcessRegressor model to predict the mean and standard deviation of the cost function at the configuration. It then calculates the expected improvement using the best cost so far.
		"""
		y_pred, y_std = gp_model.predict(x, return_std=True)
		z = (y_pred - best_y) / y_std
		ei = (y_pred - best_y) * self.norm.cdf(z) + y_std * self.norm.pdf(z)
		return ei

	def __init__(self,hot_start=10,cache_length=416):
		"""
		This is the constructor of the class. It initializes the class with the following parameters:
		hot_start: if set to boolean True, does a full gridsearch. If set to an integer, does a random search for that many configurations
		cache_length: number of inferences to cache before updating the history
		"""
		self.optimizer_queue = []
		self.hot_start = hot_start
		if self.hot_start == 0:
			self.hot_start = 1
		
		super().__init__(cache_length=cache_length)
		
		# # overwrite the history data structure. We will use multiple lists to store the history
		# self.history = [] # each element is a configuration
		# self.history_cost = [] # each element is the cost of the configuration
		# self.history_optim_T = [] # each element is the optim_T of the configuration

		# overwrite the history data structure. We will use a dictionary to store the history
		self.history = {} # each key is a configuration. each value is the cost of the configuration
		self.history_optim_T = {} # each key is a configuration. each value is the optim_T of the configuration

		self.inp_domain = [(x,y,z) for x in self.cpu_values if x >= self.cpu_min_limit \
					 					for y in self.gpu_values if y >= self.gpu_min_limit \
										for z in self.batchsize_values if z >= self.batchsize_min_limit]
		self.gp_model = None
	
	def update_history(self):
		"""
		This function is used to update the history data structure. It calculates the cost of the last configuration and updates the history matrix. This is different from the base class because it uses a dictionary to store the history
		"""
		# calculate the cost of the last configuration
		mean_energy = np.median(self.cache_energy[100:])
		mean_time = np.median(self.cache_time[100:])
		
		try:
			cost = self.alpha*mean_energy/self.energy_baseline + (1-self.alpha)*mean_time/self.time_baseline
		except:
			cost = float("inf")
		
		print("last set config",self.last_set_config,"cost",cost, "last set config comment", self.last_set_config_comment)
		
		# self.history.append(self.last_set_config)
		# self.history_cost.append(cost)
		# self.history_optim_T.append(self.optim_T)

		self.history[self.last_set_config] = cost
		self.history_optim_T[self.last_set_config] = self.optim_T
		self.save_logs_optim(mean_energy, mean_time, cost)
		self.cache_energy = []
		self.cache_time = []

	def run_optimizer(self):
		"""
		This function is used to run the optimizer. It is called after every cache_length inferences. It uses bayesian optimization to find the best configuration. It uses the GaussianProcessRegressor model to predict the mean and standard deviation of the cost function at the configuration. It then calculates the expected improvement using the best cost so far.
		"""
		if len(self.optimizer_queue) > 0:
			self.set_config(*self.optimizer_queue.pop(0), comment="queue pop")
			return

		# if hot_start is set to boolean True, do a full gridsearch
		if self.hot_start > 0:
			self.optimizer_queue = [self.inp_domain[i] for i in np.random.choice(len(self.inp_domain), self.hot_start)]
			self.hot_start = 0
			self.set_config(*self.optimizer_queue.pop(0), comment="random search")
			return
		
		if self.gp_model is None:
			# if the model is not yet initialized, initialize it with the known data
			kernel = 1.0 * self.RBF(length_scale=1.0)
			self.gp_model = self.GaussianProcessRegressor(kernel=kernel)
		
		X = np.array(list(self.history.keys()))
		y = np.array(list(self.history.values())) * -1 # we want to minimize the cost
		
		self.gp_model.fit(X, y)
		best_y = np.max(y)
		ei = self.expected_improvement(self.inp_domain, self.gp_model, best_y)
		best_config = self.inp_domain[np.argmax(ei)]

		self.set_config(*best_config, comment="bayesian optimization")
		return

optimizer = None # will be overwritten by the user


# this is used for stopping the queue service thread when needed
KEEP_RUNNING_SERVICE_THREAD = [True]

# this is used to queue the requests for the optimizer
request_queue = []
request_queue_ts = []

# user can define a function that will be used to process the batch
user_function = None

def queue_add(in_f):
	"""
	queue_add is a decorator that is used to define the function that will be used to process the batch, get requests from user_function calls and also get arrival timestamps of the requests.
	"""
	def wrapper(user_request):
		request_queue_ts.append(time.time())
		request_queue.append(user_request)
		global user_function
		user_function = in_f
		return
	return wrapper

def queue_servicing_thread():
	"""
	This function is used to service the queue. It is a thread that runs in the background and services the queue. It loads the batch, copies it to the GPU, processes the batch and gets the metrics. It then posts the results to the optimizer. It also checks if there is a significant change in the arrival rate and posts the results to the optimizer.
	"""
	while KEEP_RUNNING_SERVICE_THREAD[0]:
		try:
			if len(request_queue) >= BATCH_SIZE:
				# load the batch
				batch = request_queue[:BATCH_SIZE]
				batch_arr_ts = request_queue_ts[:BATCH_SIZE]
				del request_queue[:BATCH_SIZE]
				del request_queue_ts[:BATCH_SIZE]

				# copy the batch to the GPU
				if optimizer.model_type == "resnet":
					batch_input = torch.cat(batch).cuda()
				if optimizer.model_type == "bert":
					if optimizer.bert_tokenizer is None:
						from transformers import BertTokenizer
						model_name = "prajjwal1/bert-tiny"
						tokenizer = BertTokenizer.from_pretrained(model_name)
						optimizer.bert_tokenizer = tokenizer
					batch_input = optimizer.bert_tokenizer(batch, padding=True, truncation=True, 
															return_tensors="pt", max_length=512).to("cuda")

				# process the batch and get the metrics
				batch_processing_start = time.time()
				energy_batch = power_profile.energy_calculator(user_function, batch_input)
				t_end = time.time()
				time_taken_i = [t_end - t for t in batch_arr_ts]

				# check if there is a significant change in the arrival rate
				optimizer.arrival_rate_observer(batch_arr_ts)

				# post the results to the optimizer. Results = energy and time taken for each inference as a list
				save_logs_tasks(batch_arr_ts,[batch_processing_start]*BATCH_SIZE,[energy_batch/BATCH_SIZE]*BATCH_SIZE, time_taken_i, optimizer.optim_T)
				optimizer.post_results([energy_batch/BATCH_SIZE]*BATCH_SIZE, time_taken_i)
				
				# print("processed batch")
		except KeyboardInterrupt:
			print("interrupted")
			break
	print("Stopping queue service thread")

global logs_tasks_fp
logs_tasks_fp = None
def save_logs_tasks(arr_ts, start_time, energy, time_taken, optim_T):
	"""
	This function is used to save the logs of the tasks. It saves the arrival timestamp, start time, energy, time taken and the optim_T of the tasks. It takes the following parameters:
	arr_ts: arrival timestamp of the tasks
	start_time: start time of the tasks
	energy: energy of the tasks
	time_taken: time taken for the tasks
	optim_T: optim_T of the tasks. It is the number of times the optimizer has been called so far
	"""
	global logs_tasks_fp, logs_text_prefix

	if logs_tasks_fp is None:
		logs_tasks_fp = open(logs_text_prefix+"logs_tasks.csv", "w")
		logs_tasks_fp.write("arr_ts,start_time,energy,time_taken,optim_T\n")

	if len(arr_ts) != len(start_time) or len(energy) != len(time_taken) or len(arr_ts) != len(energy):
		print("lengths",len(arr_ts),len(start_time),len(energy),len(time_taken))
		print("something went wrong in saving logs")
		return
	for i in range(len(energy)):
		logs_tasks_fp.write(f"{arr_ts[i]},{start_time[i]},{energy[i]},{time_taken[i]},{optim_T}\n")

def queue_service_stop():
	"""
	This function is used to stop the queue service thread. It sets the global variable KEEP_RUNNING_SERVICE_THREAD to False and waits for the thread to stop.
	"""
	time.sleep(60) # wait for the queue to be empty
	KEEP_RUNNING_SERVICE_THREAD[0] = False
	t.join(timeout=60) # wait for the service thread to stop
	optimizer.optimizer_stop()
	if logs_tasks_fp is not None:
		logs_tasks_fp.close()

# launch the queue service thread
t = threading.Thread(target=queue_servicing_thread)
t.start()
