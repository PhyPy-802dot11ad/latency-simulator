"""
Data generators and simulation invocation functions.
"""


import os
import sys
import pickle
from datetime import datetime as dt

import simpy
import numpy as np
from tqdm import tqdm

# from Helpers import calc_blocked_payload_size_and_num_of_blocks, get_demapper_delay_instance
from .Helpers import calc_block_params, get_demapper_delay_instance, calc_cw_params
# from .RxDbb import RxDbb_Flavour3Ideal, RxDbb_Flavour2, RxDbbSimplified448


STF_LENGTH = 2176
CES_LENGTH = 1152
HEADER_LENGTH = 1024
BLOCK_LENGTH = 448
GI_LENGTH = 64
BLOCK_AND_GI_LENGTH = 512
SAMPLING_RATE_GHZ = 1.76


def sender(env, rx_dbb, mpdu_len_symbols, header_part_of_payload=False):
    """Generate and yield transmission sequences. STF and CES are on a per-block basis, payload is symbol-wise.

    :param env: SimPy environment
    :param rx_dbb: Receiver digital baseband object
    :param mpdu_len: Payload length in symbols (including GIs)
    :param header_part_of_payload: Flag indicating whether the header delay is invocated separately (when false)
    """

    # STF arrival
    yield env.timeout(STF_LENGTH / SAMPLING_RATE_GHZ)
    rx_dbb.STF_ingress_block.input_data_buffer.put( [None] )

    # CES arrival
    yield env.timeout(CES_LENGTH / SAMPLING_RATE_GHZ)
    rx_dbb.CES_ingress_block.input_data_buffer.put( [None] )

    # Header arrival
    if not header_part_of_payload:
        yield env.timeout(HEADER_LENGTH / SAMPLING_RATE_GHZ)

    # symbol = np.zeros(1, dtype=complex)
    symbol = np.zeros(1, dtype=np.uint8)

    # Payload arrival
    for k in tqdm(range(mpdu_len_symbols), position=0, leave=True):

        yield env.timeout(1 / SAMPLING_RATE_GHZ)
        # rx_dbb.PAY_ingress_block.input_data_buffer.put(np.array([symbol]))
        rx_dbb.PAY_ingress_block.input_data_buffer.put(symbol)

    # Last GI delay (FDE and other blocks are based on block + succeeding GI)
    yield env.timeout(GI_LENGTH / SAMPLING_RATE_GHZ)


def sender_simplified(env, rx_dbb, num_of_data_blocks, header_part_of_payload=False):
    """Generate and yield transmission sequences.
    Sends entire data blocks to reduce the number of events and speed up execution times

    :param env: SimPy environment
    :param rx_dbb: Receiver digital baseband object
    :param num_of_data_blocks: Number of data blocks (448 + 64 symbols)
    :param header_part_of_payload: Flag indicating whether the header delay is invocated separately (when false)
    """

    # STF arrival
    yield env.timeout(STF_LENGTH / SAMPLING_RATE_GHZ)

    # CES arrival
    yield env.timeout(CES_LENGTH / SAMPLING_RATE_GHZ)
    rx_dbb.CES_ingress_block.input_data_buffer.put( [None] )

    # Header (no fruther delays taken into account)
    if not header_part_of_payload:
        yield env.timeout(HEADER_LENGTH / SAMPLING_RATE_GHZ)

    symbol_block = np.zeros(BLOCK_AND_GI_LENGTH)

    # Payload arrival
    for k in range(num_of_data_blocks):

        print(f'({dt.now()}) | Sending block {k} of {num_of_data_blocks}.')
        sys.stdout.flush()

        yield env.timeout(BLOCK_AND_GI_LENGTH / SAMPLING_RATE_GHZ)
        rx_dbb.PAY_ingress_block.input_data_buffer.put(symbol_block)

    # Last GI delay (FDE and other blocks are based on block + succeeding GI)
    yield env.timeout(GI_LENGTH / SAMPLING_RATE_GHZ)


# def simulate( log_path, MCS, decoder_iterations, demapping_algorithm, payload_length, aggregated_ppdus ):
def simulate( out_path, rx_dbb_reference, MCS, decoder_iterations, demapping_algorithm, payload_length_bits, simplified ):
    """Start simulation using RX DBB that processes the payload on a per-symbol basis.

    :param out_path: Absolute path where the logs are stored
    :param rx_dbb_reference: Class definition of the RX DBB used in the simulation
    :param MCS: Selected MCS
    :param decoder_iterations: Number of LDPC decoder iterations during simulation
    :param demapping_algorithm: Demapping algorithm name, used to spawn demapper instance
    :param payload_length_bits: Lenght of the payload in bits
    :param simplified: Indication to use simplified sender
    """

    time_started = dt.now()

    # log_subdir = f'{out_id:04}'
    log_subdir = f'{MCS}_{demapping_algorithm}_{decoder_iterations}'

    if not os.path.exists( os.path.join( out_path, log_subdir) ):
        os.mkdir(os.path.join( out_path, log_subdir ))

    sys.stdout = open( os.path.join( out_path, log_subdir, 'std.out' ), 'w' )

    demapper_delay_instance = get_demapper_delay_instance(demapping_algorithm)


    env = simpy.Environment()

    num_of_codewords, codeword_padding = calc_cw_params( MCS, payload_length_bits )
    blocked_payload_length, num_of_blocks, block_padding = calc_block_params( MCS, payload_length_bits )

    rx_dbb = rx_dbb_reference(
        out_path,
        log_subdir,
        env,
        MCS,
        num_of_blocks,
        block_padding,
        num_of_codewords,
        codeword_padding,
        decoder_iterations,
        demapper_delay_instance
    )

    # Activate blocks
    [ env.process(block.run()) for block in rx_dbb.blocks ]

    # Start sending
    if simplified:
        env.process(sender_simplified(env, rx_dbb, num_of_blocks))
    else:
        env.process(sender(env, rx_dbb, blocked_payload_length))


    # Run
    env.run()

    # Store the buffer logs and the corresponding block descriptions
    for block in tqdm(rx_dbb.blocks):

        block.save_logs()

        # name = '_'.join(block.name.split(' '))
        name = '_'.join(block.name.split(' ')).lower()
        # name = block.name.lower()

        # After logs in order for the destination dir to exist already
        # path = os.path.join(log_path, log_subdir, f'{block.block_id:02d}-{name}-Description.pkl' )
        path = os.path.join(out_path, log_subdir, f'{block.block_id:02d}-{name}-description.pkl' )
        # path = os.path.join(log_subdir, f'{block.block_id:02d}-{name}-Description.pkl' )
        with open(path, 'wb') as f:
            pickle.dump(block.describe(), f)

    # path = os.path.join(log_path, log_subdir, 'Metadata.pkl')
    path = os.path.join(out_path, log_subdir, 'metadata.pkl')
    metadata = dict(
        MCS=MCS,
        decoder_iterations=decoder_iterations,
        demapping_algorithm=demapping_algorithm,
        time_started=time_started,
        time_ended=dt.now()
    )
    with open(path, 'wb') as f: pickle.dump(metadata, f)

    # Control lines, showing when execution has stopped
    print(metadata['time_started'])
    print(metadata['time_ended'])
    sys.stdout.flush()


def sender_simplified(env, rx_dbb, num_of_data_blocks, header_part_of_payload=False):
    """Generate and yield transmission sequences.
    Sends entire data blocks to reduce the number of events and speed up execution times

    :param env: SimPy environment
    :param rx_dbb: Receiver digital baseband object
    :param num_of_data_blocks: Number of data blocks (448 + 64 symbols)
    :param header_part_of_payload: Flag indicating whether the header delay is invocated separately (when false)
    """

    # STF arrival
    yield env.timeout(STF_LENGTH / SAMPLING_RATE_GHZ)

    # CES arrival
    yield env.timeout(CES_LENGTH / SAMPLING_RATE_GHZ)
    rx_dbb.CES_ingress_block.input_data_buffer.put( [None] )

    # Header (no fruther delays taken into account)
    if not header_part_of_payload:
        yield env.timeout(HEADER_LENGTH / SAMPLING_RATE_GHZ)

    symbol_block = np.zeros(BLOCK_AND_GI_LENGTH)

    # Payload arrival
    for k in range(num_of_data_blocks):

        print(f'({dt.now()}) | Sending block {k} of {num_of_data_blocks}.')
        sys.stdout.flush()

        yield env.timeout(BLOCK_AND_GI_LENGTH / SAMPLING_RATE_GHZ)
        rx_dbb.PAY_ingress_block.input_data_buffer.put(symbol_block)

    # Last GI delay (FDE and other blocks are based on block + succeeding GI)
    yield env.timeout(GI_LENGTH / SAMPLING_RATE_GHZ)


# def simulate_simplified( out_path, rx_dbb_reference, MCS, decoder_iterations, demapping_algorithm, payload_length_bits ):
#     """Start simulation using simplified RX DBB (non-generic blocks, delays on a per-block basis).
#
#     :param out_path: Absolute path where the logs are stored.
#     :param rx_dbb_reference: Class definition of the RX DBB used in the simulation.
#     :param MCS: Selected MCS.
#     :param decoder_iterations: Number of LDPC decoder iterations during simulation.
#     :param demapping_algorithm: Demapping algorithm name, used to spawn demapper instance.
#     :param payload_length_bits: Lenght of the payload in bits.
#     """
#
#     time_started = dt.now()
#
#     # log_subdir = f'{out_id:04}'
#     log_subdir = f'{MCS}_{demapping_algorithm}_{decoder_iterations}'
#
#     if not os.path.exists( os.path.join( out_path, log_subdir) ):
#         os.mkdir(os.path.join( out_path, log_subdir ))
#
#     sys.stdout = open( os.path.join( out_path, log_subdir, 'std.out' ), 'w' )
#
#     demapper_delay_instance = get_demapper_delay_instance(demapping_algorithm)
#
#     env = simpy.Environment()
#
#     num_of_codewords, codeword_padding = calc_cw_params( MCS, payload_length_bits )
#     blocked_payload_length, num_of_blocks, block_padding = calc_block_params( MCS, payload_length_bits )
#
#     rx_dbb = rx_dbb_reference(
#         out_path,
#         log_subdir,
#         env,
#         MCS,
#         num_of_blocks,
#         block_padding,
#         num_of_codewords,
#         codeword_padding,
#         decoder_iterations,
#         demapper_delay_instance
#     )
#
#     # Activate blocks
#     [ env.process(block.run()) for block in rx_dbb.blocks ]
#
#     # Start sending
#     env.process(sender_simplified(env, rx_dbb, num_of_blocks))
#
#     # Run
#     env.run()
#
#     # Store the buffer logs and the corresponding block descriptions
#     for block in rx_dbb.blocks:
#
#         block.save_logs()
#
#         name = '_'.join(block.name.split(' ')).lower() # Remove spaces and convert to lowercase
#
#         # After logs in order for the destination dir to exist already
#         path = os.path.join(out_path, log_subdir, f'{block.block_id:02d}-{name}-description.pkl' )
#         # path = os.path.join(log_subdir, f'{block.block_id:02d}-{name}-Description.pkl' )
#         with open(path, 'wb') as f:
#             pickle.dump(block.describe(), f)
#
#     path = os.path.join(out_path, log_subdir, 'metadata.pkl')
#     metadata = dict(
#         MCS=MCS,
#         decoder_iterations=decoder_iterations,
#         demapping_algorithm=demapping_algorithm,
#         payload_length_bits=payload_length_bits,
#         time_started=time_started,
#         time_ended=dt.now()
#     )
#     with open(path, 'wb') as f: pickle.dump(metadata, f)
#
#     # Control lines, showing when execution has stopped
#     print(metadata['time_started'])
#     print(metadata['time_ended'])
#     sys.stdout.flush()
