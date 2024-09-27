import time
import threading
import os
import json

LOGGING_INTERVAL_s = 0.010


try:
	with open("/sys/module/tegra_fuse/parameters/tegra_chip_id") as f:
		board = int(f.read().strip())
except FileNotFoundError:
	with open("/sys/devices/soc0/soc_id") as f:
		board = int(f.read().strip())

board_convert =	{
					25:"Xavier",
					24:"TX2",
					33:"TX1/Nano",
					64:"TK1",
				}

board = board_convert[board]
print("running on Jetson",board)

jetpack_type = 0

if board == "TX1/Nano":
	i2c_folder = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/"

elif board=="Xavier":
	i2c_folder = "/sys/bus/i2c/drivers/ina3221x/7-0040/iio:device0/"
	if not os.path.exists(i2c_folder):
		i2c_folder = "/sys/bus/i2c/devices/7-0040/hwmon/hwmon5/"
		jetpack_type = 1

else:
	print("Board not supported")
	exit()

print("jetpack_type",jetpack_type)

def logging(data_passthrough):
	try:
		total_power = 0
		while True:
			if jetpack_type == 0:
				f_INP_power1 = open(i2c_folder+"in_power0_input","r")
				# f_INP_power2 = open(i2c_folder+"in_power1_input","r")
				# f_INP_power3 = open(i2c_folder+"in_power2_input","r")
				total_power += int(f_INP_power1.read())
			elif jetpack_type == 1:
				f_INP_voltage1 = open(i2c_folder+"in1_input","r")
				f_INP_current1 = open(i2c_folder+"curr1_input","r")
				INP_POWER = (int(f_INP_voltage1.read())*int(f_INP_current1.read()))/1000000
				total_power += INP_POWER
				
			time.sleep(LOGGING_INTERVAL_s)

			if data_passthrough[0]:
				total_energy = total_power*LOGGING_INTERVAL_s
				data_passthrough[0] = total_energy
				break

	except KeyboardInterrupt:
		pass
	
def energy_calculator(input_function, function_param):
	data_passthrough = [None]
	t = threading.Thread(target=logging, args=(data_passthrough,))
	t.start()
	input_function(function_param)
	data_passthrough[0] = True
	t.join()
	return data_passthrough[0]
