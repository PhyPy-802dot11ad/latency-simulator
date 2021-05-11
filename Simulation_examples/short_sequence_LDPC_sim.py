"""
Simulate the transmission of short payload sequences (100B) for all supported MCSs and 1-25 LDPC decoding iterations.

As used for obtaining the results presented at WIC SITB 2021.
"""


import os
from datetime import datetime as dt
from multiprocessing import Pool

import numpy as np

from PHY_802dot11ad_latency_sim import simulate, simulate_simplified, RxDbbSimplified448, RxDbb_Flavour2


# TODO: check that all params are reverted back to those used in WIC paper


### Const ##############################################################################################################

MAX_MPDU_LENGTH_B = 262_143


### Config #############################################################################################################

### Set process-related params
os.nice(10) # Be nice to other processes, allow preemption
POOL_SIZE = 8 # Set max number of concurrent processes

### Set output log path
log_basepath = os.path.dirname(__file__)

### Select normal vs. fast simulation
simulation_function = simulate
# simplified_simulation = False
# rx_dbb_reference = RxDbb_Flavour2
simplified_simulation = True
rx_dbb_reference = RxDbbSimplified448
# simulation_function = simulate_simplified

### Set simulation arguments
# MPDU_length_bytes = MAX_MPDU_LENGTH_B
MPDU_length_bytes = 100

mcs_array = np.concatenate((
    np.arange(2, 12+1, 1, dtype=float),
    np.array([12.1, 12.3, 12.4, 12.5])
))

iterations_array = np.arange(1, 100.25, 0.25)

demapping_alg_array = np.array(['decision threshold']) # 'optimal', 'suboptimal', 'decision threshold'


### Run ################################################################################################################

# Allocate storage for log
start_time = dt.now().isoformat()
out_path = os.path.join( log_basepath, 'Log', start_time)
os.mkdir( out_path )

np.random.seed(0) # Reset random

payload_length_bits = MPDU_length_bytes * 8

# Define all possible MCS, iteration, and demapping algorithm combinations
combination_list = []
for mcs in mcs_array:
    for demapping_alg in demapping_alg_array:
        for iterations in iterations_array:
            combination_list.append((mcs, iterations, demapping_alg))

print(f'Starting simultaions for {len(combination_list)} combinations.')

# Translate the possible combinations into simulation arguments
sim_args = []
for mcs, decoder_iterations, demapper_delay_instance in combination_list:
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
    p.starmap(simulation_function, sim_args)

# # Run in single-process for debugging purposes
# for sa in sim_args:
#     simulation_function(*sa)
