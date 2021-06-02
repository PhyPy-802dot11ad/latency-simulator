"""
Simulate transmission of payload sequences ranging 1 to 262 KB in length.
"""

import os
from datetime import datetime as dt
from multiprocessing import Pool
import itertools

import numpy as np

from PHY_802dot11ad_latency_sim import simulate, RxDbbScFde, RxDbbScFdeSimplified448


### Const ##############################################################################################################

MAX_MPDU_LENGTH_B = 262_143
MAX_MPDU_LENGTH_KB = 262


### Config #############################################################################################################

### Set process-related params
os.nice(10) # Be nice to other processes, allow preemption
POOL_SIZE = 8 # Set max number of concurrent processes

### Set output log path
log_basepath = os.path.dirname(__file__)

### Select normal vs. fast simulation
# rx_dbb_reference = RxDbbScFde; simplified_simulation = False
rx_dbb_reference = RxDbbScFdeSimplified448; simplified_simulation = True

### Set simulation arguments
MPDU_length_bytes = np.arange(1,MAX_MPDU_LENGTH_KB+1,1) * 1000
# MPDU_length_bytes = np.arange(1,10,1)

mcs_array = np.concatenate((
    np.arange(2, 12+1, 1, dtype=float),
    np.array([12.1, 12.3, 12.4, 12.5])
))

iterations_array = np.array([1,10])

demapping_alg_array = np.array(['decision threshold']) # 'optimal', 'suboptimal', 'decision threshold'


### Run ################################################################################################################

# Allocate storage for log
start_time = dt.now().isoformat()
out_path = os.path.join( log_basepath, 'Log', start_time)
os.mkdir( out_path )

np.random.seed(0) # Reset random

payload_length_bits = MPDU_length_bytes * 8

# Define all possible combinations
combinations = itertools.product(payload_length_bits, mcs_array, iterations_array, demapping_alg_array)

_len = payload_length_bits.size * mcs_array.size * iterations_array.size * demapping_alg_array.size
print(f'Starting simultaions for {_len} combinations.')

# Translate the possible combinations into simulation arguments
sim_args = []
# for payload_length_bits, mcs, decoder_iterations, demapper_delay_instance in combination_list:
for payload_length_bits, mcs, decoder_iterations, demapper_delay_instance in combinations:
    tmp_args = [
        out_path,
        rx_dbb_reference,
        mcs,
        decoder_iterations,
        demapper_delay_instance,
        payload_length_bits,
        simplified_simulation
    ]
    sim_args.append(tmp_args)

# Run processes from pool using the pre-generated arguments
with Pool(POOL_SIZE) as p:
    p.starmap(simulate, sim_args)
