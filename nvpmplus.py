import argparse
import os
import subprocess
import time


try:
    with open("/sys/module/tegra_fuse/parameters/tegra_chip_id") as f:
        board = int(f.read().strip())
except FileNotFoundError:
    try:
        with open("/sys/devices/soc0/soc_id") as f:
            board = int(f.read().strip())
    except FileNotFoundError:
        board = None
        print("Unknown board")

board_convert = {
    25: "Xavier",
    24: "TX2",
    33: "TX1/Nano",
    64: "TK1",
    35: "Orin",
}

board = board_convert[board]
print("running on Jetson", board)

if board == "TX1/Nano":
    print("WARNING!!! NOT FULLY TESTED ON TX1/Nano")
    cpu_scaling_available_frequencies = [
        102000,
        204000,
        307200,
        403200,
        518400,
        614400,
        710400,
        825600,
        921600,
        1036800,
        1132800,
        1224000,
        1326000,
        1428000,
        1479000,
    ]

    gpu_available_frequencies = [
        76800000,
        153600000,
        230400000,
        307200000,
        384000000,
        460800000,
        537600000,
        614400000,
        691200000,
        768000000,
        844800000,
        921600000,
    ]
    gpu_loc = "/sys/devices/gpu.0/devfreq/57000000.gpu/"
    cpu_lim = 4

elif board == "Xavier":
    cpu_scaling_available_frequencies = [
        115200,
        192000,
        268800,
        345600,
        422400,
        499200,
        576000,
        652800,
        729600,
        806400,
        883200,
        960000,
        1036800,
        1113600,
        1190400,
        1267200,
        1344000,
        1420800,
        1497600,
        1574400,
        1651200,
        1728000,
        1804800,
        1881600,
        1907200,
    ]

    gpu_available_frequencies = [
        114750000,
        204000000,
        306000000,
        408000000,
        510000000,
        599250000,
        701250000,
        752250000,
        803250000,
        854250000,
        905250000,
        956250000,
        1007250000,
        1058250000,
        1109250000,
    ]
    gpu_loc = "/sys/devices/gpu.0/devfreq/17000000.gv11b/"
    cpu_lim = 6

elif board == "Orin":
    # Orin series frequencies - actual frequencies from the system
    cpu_scaling_available_frequencies = [
        115200,
        192000,
        268800,
        345600,
        422400,
        499200,
        576000,
        652800,
        729600,
        806400,
        883200,
        960000,
        1036800,
        1113600,
        1190400,
        1267200,
        1344000,
        1420800,
        1497600,
        1574400,
        1651200,
        1728000,
        1804800,
        1881600,
        1958400,
        2035200,
        2112000,
        2188800,
        2201600,
    ]

    gpu_available_frequencies = [
        306000000,
        408000000,
        510000000,
        612000000,
        714000000,
        816000000,
        918000000,
        1020000000,
        1122000000,
        1224000000,
        1300500000,
    ]
    gpu_loc = "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/"
    cpu_lim = 8  # Orin typically has 8 cores

else:
    print("Board not supported")
    exit()

# WRITE_WAIT_TIME = 0.001

cpu_govs = [
    "interactive",
    "conservative",
    "ondemand",
    "userspace",
    "powersave",
    "performance",
    "schedutil",
]
gpu_govs = [
    "wmark_simple",
    "nvhost_podgov",
    "userspace",
    "performance",
    "simple_ondemand",
]


def set_state(cpus, cpu_max_fq, gpu_max_fq):
    # subprocess.Popen("sudo nvpmodel -m 0",stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,shell=True)
    # time.sleep(1)
    # subprocess.Popen("sudo nvpmodel -m 0",stdin=subprocess.PIPE,shell=True)
    # time.sleep(1)

    # convert neg input to positive
    if cpu_max_fq < 0:
        cpu_max_fq = len(cpu_scaling_available_frequencies) + cpu_max_fq
    if gpu_max_fq < 0:
        gpu_max_fq = len(gpu_available_frequencies) + gpu_max_fq

    if os.getuid() != 0:
        raise Exception(
            "This program is not run as sudo or elevated this it will not work"
        )

    for i in range(cpu_lim):
        filename = "/sys/devices/system/cpu/cpu{}/online".format(i)
        state = int(cpus > i)
        # print(filename,state)
        with open(filename, "w") as f:
            f.write(str(state))
    # time.sleep(WRITE_WAIT_TIME)

    # get current freqs
    with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq") as f:
        cpu_min_curr_fq = int(f.read().strip())
        cpu_min_curr_fq = cpu_scaling_available_frequencies.index(cpu_min_curr_fq)
    with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq") as f:
        cpu_max_curr_fq = int(f.read().strip())
        cpu_max_curr_fq = cpu_scaling_available_frequencies.index(cpu_max_curr_fq)

    # print("current",cpu_min_curr_fq,cpu_max_curr_fq)
    # print("new",cpu_max_fq,gpu_max_fq)

    if cpu_max_fq < cpu_min_curr_fq:
        filename = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
        state = cpu_scaling_available_frequencies[cpu_max_fq]
        # print(filename,state)
        with open(filename, "w") as f:
            f.write(str(state))
            # time.sleep(WRITE_WAIT_TIME)

        filename = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"
        state = cpu_scaling_available_frequencies[cpu_max_fq]
        # print(filename,state)
        with open(filename, "w") as f:
            f.write(str(state))
            # time.sleep(WRITE_WAIT_TIME)

    else:
        filename = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"
        state = cpu_scaling_available_frequencies[cpu_max_fq]
        # print(filename,state)
        with open(filename, "w") as f:
            f.write(str(state))
            # time.sleep(WRITE_WAIT_TIME)

        filename = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
        state = cpu_scaling_available_frequencies[cpu_max_fq]
        # print(filename,state)
        with open(filename, "w") as f:
            f.write(str(state))
            # time.sleep(WRITE_WAIT_TIME)

    # get current freqs
    with open(gpu_loc + "min_freq") as f:
        gpu_min_curr_fq = int(f.read().strip())
        gpu_min_curr_fq = gpu_available_frequencies.index(gpu_min_curr_fq)
    with open(gpu_loc + "max_freq") as f:
        gpu_max_curr_fq = int(f.read().strip())
        gpu_max_curr_fq = gpu_available_frequencies.index(gpu_max_curr_fq)

    if gpu_max_fq < gpu_min_curr_fq:
        filename = gpu_loc + "min_freq"
        state = gpu_available_frequencies[gpu_max_fq]
        # print(filename,state)
        with open(filename, "w") as f:
            f.write(str(state))
            # time.sleep(WRITE_WAIT_TIME)

        filename = gpu_loc + "max_freq"
        state = gpu_available_frequencies[gpu_max_fq]
        # print(filename,state)
        with open(filename, "w") as f:
            f.write(str(state))
            # time.sleep(WRITE_WAIT_TIME)
    else:
        filename = gpu_loc + "max_freq"
        state = gpu_available_frequencies[gpu_max_fq]
        # print(filename,state)
        with open(filename, "w") as f:
            f.write(str(state))
            # time.sleep(WRITE_WAIT_TIME)

        filename = gpu_loc + "min_freq"
        state = gpu_available_frequencies[gpu_max_fq]
        # print(filename,state)
        with open(filename, "w") as f:
            f.write(str(state))
            # time.sleep(WRITE_WAIT_TIME)


def read_state():
    with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq") as f:
        cpu_min_curr_fq = int(f.read().strip())
        cpu_min_curr_fq = cpu_scaling_available_frequencies.index(cpu_min_curr_fq)
    with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq") as f:
        cpu_max_curr_fq = int(f.read().strip())
        cpu_max_curr_fq = cpu_scaling_available_frequencies.index(cpu_max_curr_fq)

    with open(gpu_loc + "min_freq") as f:
        gpu_min_curr_fq = int(f.read().strip())
        gpu_min_curr_fq = gpu_available_frequencies.index(gpu_min_curr_fq)
    with open(gpu_loc + "max_freq") as f:
        gpu_max_curr_fq = int(f.read().strip())
        gpu_max_curr_fq = gpu_available_frequencies.index(gpu_max_curr_fq)

    return cpu_min_curr_fq, cpu_max_curr_fq, gpu_min_curr_fq, gpu_max_curr_fq


def set_gov(cpu_gov, gpu_gov):
    cpu_govs = [
        "interactive",
        "conservative",
        "ondemand",
        "userspace",
        "powersave",
        "performance",
        "schedutil",
    ]
    gpu_govs = [
        "wmark_simple",
        "nvhost_podgov",
        "userspace",
        "performance",
        "simple_ondemand",
    ]

    print(
        "WARNING!!! Changing governor is not fully optimized. Code cleaning is needed"
    )

    subprocess.Popen(
        "sudo nvpmodel -m 8",
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True,
    )
    time.sleep(1)
    subprocess.Popen("sudo nvpmodel -m 8", stdin=subprocess.PIPE, shell=True)
    time.sleep(0.5)

    if cpu_gov not in cpu_govs or gpu_gov not in gpu_govs:
        raise Exception(
            "Not a valid governor\nCPU:" + str(cpu_govs) + "\nGPU:" + str(gpu_govs)
        )

    for i in range(cpu_lim):
        filename = "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor".format(i)
        # print(filename,cpu_gov)
        with open(filename, "w") as f:
            f.write(cpu_gov)
        # time.sleep(WRITE_WAIT_TIME)

    filename = gpu_loc + "governor"
    # print(filename,gpu_gov)
    with open(filename, "w") as f:
        f.write(gpu_gov)
    # time.sleep(WRITE_WAIT_TIME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nvpmodel plus")
    if board == "TX1/Nano":
        parser.add_argument(
            "--cpus", action="store", type=int, default=4, help="input can be 1 to 4"
        )
        parser.add_argument(
            "--cpu_max_fq",
            action="store",
            type=int,
            default=-1,
            help="input can be 0 to 14",
        )
        parser.add_argument(
            "--gpu_max_fq",
            action="store",
            type=int,
            default=-1,
            help="input can be 0 to 11",
        )
    elif board == "Xavier":
        parser.add_argument(
            "--cpus", action="store", type=int, default=6, help="input can be 1 to 6"
        )
        parser.add_argument(
            "--cpu_max_fq",
            action="store",
            type=int,
            default=-1,
            help="input can be 0 to 24",
        )
        parser.add_argument(
            "--gpu_max_fq",
            action="store",
            type=int,
            default=-1,
            help="input can be 0 to 14",
        )
    elif board == "Orin":
        parser.add_argument(
            "--cpus", action="store", type=int, default=8, help="input can be 1 to 8"
        )
        parser.add_argument(
            "--cpu_max_fq",
            action="store",
            type=int,
            default=-1,
            help="input can be 0 to 28",
        )
        parser.add_argument(
            "--gpu_max_fq",
            action="store",
            type=int,
            default=-1,
            help="input can be 0 to 10",
        )

    parser.add_argument(
        "--cpu_gov",
        action="store",
        type=str,
        default="schedutil",
        help="input can be a cpu governor. \
					 					Possible values are: interactive, conservative, ondemand, userspace, powersave, performance, schedutil",
    )
    parser.add_argument(
        "--gpu_gov",
        action="store",
        type=str,
        default="nvhost_podgov",
        help="input can be a gpu governor. \
					 					Possible values are: wmark_simple, nvhost_podgov, userspace, performance, simple_ondemand",
    )
    parser.add_argument(
        "--ONLY_GOV",
        action="store",
        type=bool,
        default=False,
        help="Set only the governor",
    )
    parser.add_argument(
        "--ONLY_FREQ",
        action="store",
        type=bool,
        default=False,
        help="Set only the frequency",
    )
    args = parser.parse_args()
    print(args)

    if args.ONLY_GOV:
        set_gov(args.cpu_gov, args.gpu_gov)
    elif args.ONLY_FREQ:
        set_state(args.cpus, args.cpu_max_fq, args.gpu_max_fq)
    else:
        set_gov(args.cpu_gov, args.gpu_gov)
        set_state(args.cpus, args.cpu_max_fq, args.gpu_max_fq)


"""
	Example usage for only setting the governor
		python3 nvpmodel_plus.py --ONLY_GOV True --cpu_gov schedutil --gpu_gov nvhost_podgov
"""
