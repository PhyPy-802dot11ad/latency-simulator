import os

import pandas as pd
import numpy as np

from .Definitions import Delay


def get_Rm_and_Rc_from_MCS(MCS):
    """Get the modulation and coding rate based on the selected MCS."""
    mcs_table_path = os.path.join(os.path.dirname(__file__), 'MCS_table.csv')
    df = pd.read_csv(mcs_table_path, index_col='MCS')
    try:
        Rm = df.loc[MCS]['Modulation_rate']
        Rc = df.loc[MCS]['Code_rate']
        return (Rm, Rc)
    except:
        raise ValueError(f"Unknonw MCS {MCS}.")

def calc_combined_padded_dw_length( Rc, payload_length ):
    """Calculate the total number of bits in all datawords, including padding."""
    # combined_padded_dw_length = int(np.ceil(payload_length / (Rc * 672)) * Rc * 672)
    combined_padded_dw_length = calc_combined_cw_length( Rc, payload_length ) * Rc
    return int(combined_padded_dw_length)

def calc_combined_cw_length( Rc, payload_length ):
    """Calculate the total number of bits in all codewords (number of cw*672)."""
    combined_cw_length = int(np.ceil(payload_length / (Rc * 672)) * 672)
    return int(combined_cw_length)

def calc_num_of_blocks_from_cw_payload(Rm, cw_padded_payload_length ):
    """Calculate the number of blocks, given the encoded payload size."""
    number_of_blocks = np.ceil((cw_padded_payload_length / Rm) / 448)
    return int(number_of_blocks)

def calc_cw_params( MCS, payload_length ):
    """Calculate the number of codewords and codeword padding"""
    Rm, Rc = get_Rm_and_Rc_from_MCS(MCS)
    number_of_codewords = int(np.ceil(payload_length / (Rc * 672)))
    codeword_padding = int((number_of_codewords * 672 * Rc) - payload_length)
    return np.array([number_of_codewords, codeword_padding])

def calc_block_params( MCS, payload_length ):
    """Calculate the blocked payload size in a single PPDU (wo aggregation), including block padding."""
    Rm, Rc = get_Rm_and_Rc_from_MCS(MCS)
    combined_cw_length = calc_combined_cw_length(Rc, payload_length)
    block_padding = calc_block_padding_length(Rm, combined_cw_length)
    num_of_blocks = calc_num_of_blocks_from_cw_payload(Rm, combined_cw_length)
    blocked_payload_size = num_of_blocks * 512 # No final GI

    return np.array([blocked_payload_size, num_of_blocks, block_padding], dtype=int)

def calc_num_of_blocks_from_blocked_payload( blocked_payload_length ):
    """Calculate the number of blocks, given the entire blocked payload size."""
    number_of_blocks = blocked_payload_length / 512
    return int(number_of_blocks)

def calc_block_padding_length( Rm, cw_padded_payload_length ):
    """Calculate block padding length in symbols, given the data including parity bits."""
    block_padding_length = (448 - (cw_padded_payload_length / Rm) % 448) % 448 # Final % turns 448 into 0
    return int(block_padding_length)

def get_demapper_delay_instance(demapping_algorithm):
    """Get the demapper delay function based on the used algorithm."""
    if demapping_algorithm == 'optimal':
        return Delay.DemapperCustomOptimal
        # return Delay.DemapperCustomOptimal448
    elif demapping_algorithm == 'suboptimal':
        return Delay.DemapperCustomSubOptimal
        # return Delay.DemapperCustomSubOptimal448
    elif demapping_algorithm == 'decision threshold':
        return Delay.DemapperCustomDecisionThreshold
        # return Delay.DemapperCustomDecisionThreshold448
    else:
        raise ValueError(f'Unknown demapping algorithm ({demapping_algorithm})')
