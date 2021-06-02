import sys
from datetime import datetime as dt

import numpy as np

from .Buffer import Buffer



class Block():
    """Base block class."""

    def __init__(self, block_id, name, env):
        """
        :param block_id: Block block_identifier.
        :param name: Component name.
        :param env: Simpy environment.
        """

        self.block_id = block_id
        self.name = name
        self.env = env
        self.logs = []


    def add_log(self, log):
        """Add log object reference to an internal list, storing it for later saving of the logs.

        :param log: Log object reference
        """

        self.logs.append(log)

    def save_logs(self):
        """Save all logs using their stored references."""

        for log in self.logs: log.save()

    def describe(self):
        """Describe the block (verbose-like).

        :return: Dictionary comprising the block's description.
        """

        description = {
            'block_id': self.block_id,
            'name': self.name
        }

        return description



class BufferedBlock(Block):
    """Block class that includes an input- and an output data buffer."""

    def __init__(self, block_id, name, env, log_path, log_subdir, output_data_buffer=None):
        """
        :param block_id: Block block_identifier.
        :param name: Component name.
        :param env: Simpy environment.
        :param log_path: Path to the parent log directory.
        :param log_subdir: Subdirectory name within the parent log directory.
        :param output_data_buffer: Typically next block's input data buffer.
        """

        super().__init__(block_id, name, env)

        name = '_'.join(name.split(' ')).lower() # Remove spaces and convert to lowercase
        log_name = f'{block_id:02}-{name}-input_data_buffer_log'
        self.input_data_buffer = Buffer(env, log_name, log_path, log_subdir)
        self.add_log(self.input_data_buffer.log)

        self.output_data_buffer = output_data_buffer


    def describe(self):
        """Describe the block (verbose-like).

        :return: Dictionary comprising the block's description.
        """
        description = super().describe()

        description['input_data_buffer_block_id'] = id(self.input_data_buffer)
        description['output_data_buffer_block_id'] = id(self.output_data_buffer)

        return description


class LatentFunctionalBlock(BufferedBlock):
    """Block class that induce a delay when moving data to the output buffer, while also modifying the data."""

    def __init__(self, block_id, name, env, log_path, log_subdir, delay, func):
        """
        :param block_id: Block block_identifier.
        :param name: Component name.
        :param env: Simpy environment.
        :param log_path: Path to the parent log directory.
        :param log_subdir: Subdirectory name within the parent log directory.
        :param delay: Block delay class.
        :param func: Block functionality class.
        """

        super().__init__(block_id, name, env, log_path, log_subdir)

        self.delay = delay
        self.func = func


    def move_to_buffer(self, buffer, sequence):
        """Move sequence to buffer and add propagation delay.

        Additional step that allows the inclusion of a propagation delay, regardless of whether it's larger or smaller
        than the throughput delay (elapsed time between two consecutive outputs in a stream).

        :param buffer: Single Buffer instance (data or flag) or a list of buffers.
        :param sequence: Single ndarray sequence being forwarded to the buffer or a list of sequences.
        """

        delay = self.delay.get_propagation_delay()
        yield self.env.timeout(delay)

        if type(buffer) is list:
            for b, s in zip(buffer, sequence):
                b.put(s)
        else:
            buffer.put(sequence)


    def describe(self):
        """Describe the block (verbose-like).

        :return: Dictionary comprising the block's description.
        """
        description = super().describe()

        description['delay'] = self.delay.describe()
        description['func'] = self.func.describe()

        return description


class FlaglessBlock(LatentFunctionalBlock):
    """Runnable block without flags."""

    def __init__(self, block_id, name, env, log_path, log_subdir, delay, func):
        """
        :param block_id: Block block_identifier.
        :param name: Component name.
        :param env: Simpy environment.
        :param log_path: Path to the parent log directory.
        :param log_subdir: Subdirectory name within the parent log directory.
        :param delay: Block delay class.
        :param func: Block functionality class.
        """

        super().__init__(block_id, name, env, log_path, log_subdir, delay, func)


    def run(self):

        while True:

            # The block waits for data to become available before resuming
            sequence = yield self.input_data_buffer.get()
            self.input_data_buffer.log_get_granted()  # Add log entry once the resource has been successfully fetched

            # The block modifies the sequence and includes a delay
            sequence = self.func.run(sequence)
            delay = self.delay.get_throughput_delay()
            yield self.env.timeout(delay)

            # The block may provblock_ide data and flag input to other blocks
            if not self.output_data_buffer is None:
                self.env.process(self.move_to_buffer(self.output_data_buffer, sequence))

    #
    # def describe(self):
    #     """Describe the block (verbose-like).
    #
    #     :return: Dictionary comprising the block's description.
    #     """
    #     description = super().describe()
    #
    #     description['func'] = self.func.describe()
    #
    #     return description



class IOFlaggedBlock(LatentFunctionalBlock):
    """Runnable block class that supports I/O flag buffer operations."""

    def __init__(self, block_id, name, env, log_path, log_subdir, delay, func, has_input_flag_buffer, output_flag_buffer=None):
        """
        :param block_id: Block block_identifier.
        :param name: Component name.
        :param env: Simpy environment.
        :param log_path: Path to the parent log directory.
        :param log_subdir: Subdirectory name within the parent log directory.
        :param delay: Block delay class.
        :param func: Block functionality class.
        :param has_input_flag_buffer: Indicator whether the component has an input flag buffer.
        :param output_flag_buffer: Typically next block's input flag buffer.
        """

        super().__init__(block_id, name, env, log_path, log_subdir, delay, func)

        if has_input_flag_buffer:
            name = '_'.join(name.split(' ')).lower() # Remove spaces and convert to lowercase
            log_name = f'{block_id:02}-{name}-input_flag_buffer_log'
            self.input_flag_buffer = Buffer(env, log_name, log_path, log_subdir)
            self.add_log(self.input_flag_buffer.log)
        else:
            self.input_flag_buffer = None

        self.output_flag_buffer = output_flag_buffer


    def run(self):

        while True:

            # Flagged blocks can only run once they receive confirmation.
            if not self.input_flag_buffer is None:
                yield self.input_flag_buffer.get()  # Wait for flag (green light)
                self.input_flag_buffer.log_get_granted()  # Only log once the resource has been successfully fetched

            # The block waits for data to become available before resuming
            sequence = yield self.input_data_buffer.get()
            self.input_data_buffer.log_get_granted()  # Add log entry once the resource has been successfully fetched

            # The block modifies the sequence and includes a delay
            sequence = self.func.run(sequence)
            delay = self.delay.get_throughput_delay()
            yield self.env.timeout(delay)

            # The block may provblock_ide data and flag input to other blocks
            if not self.output_data_buffer is None:
                self.env.process(self.move_to_buffer(self.output_data_buffer, sequence))
            if not self.output_flag_buffer is None:
                self.env.process(self.move_to_buffer(self.output_flag_buffer, True))


    def describe(self):
        """Describe the block (verbose-like).

        :return: Dictionary comprising the block's description.
        """
        description = super().describe()

        description['input_flag_buffer_block_id'] = id(self.input_flag_buffer)
        description['output_flag_buffer_block_id'] = id(self.output_flag_buffer)

        return description



class LoopbackFlaggedBlock(LatentFunctionalBlock):
    """Runnable block class that supports I/O flag buffer operations and is self-flagged (flag output = input)."""

    def __init__(self, block_id, name, env, log_path, log_subdir, delay, func, activation_seq_len, output_flag_buffer=None):
        """
        :param block_id: Block block_identifier.
        :param name: Component name.
        :param env: Simpy environment.
        :param log_path: Path to the parent log directory.
        :param log_subdir: Subdirectory name within the parent log directory.
        :param delay: Block delay class.
        :param func: Block functionality class.
        :param activation_seq_len: Length of sequence for self-activation (self-act. duration).
        :param output_flag_buffer: Typically next block's input flag buffer.
        """

        super().__init__(block_id, name, env, log_path, log_subdir, delay, func)
        self.activation_seq_len = activation_seq_len
        self.sequences_processed = 0

        name = '_'.join(name.split(' ')).lower() # Remove spaces and convert to lowercase
        log_name = f'{block_id:02}-{name}-input_flag_buffer_log'
        self.input_flag_buffer = Buffer(env, log_name, log_path, log_subdir)
        self.add_log(self.input_flag_buffer.log)

        self.output_flag_buffer = output_flag_buffer


    def run(self):
        """Run self-activated RX baseband component instance.
        """

        while True:

            # Flagged blocks can only run once they receive confirmation.
            yield self.input_flag_buffer.get() # Wait for flag (green light)
            self.input_flag_buffer.log_get_granted() # Only log once the resource has been successfully fetched

            # The block waits for data to become available before resuming
            sequence = yield self.input_data_buffer.get()
            self.input_data_buffer.log_get_granted() # Only log once the resource has been fetched

            # The block modifies the sequence and includes a delay
            sequence = self.func.run(sequence)
            delay = self.delay.get_throughput_delay()
            yield self.env.timeout(delay)

            # The block may provblock_ide data input to other blocks
            if not self.output_data_buffer is None:
                self.env.process(self.move_to_buffer(self.output_data_buffer, sequence))

            # The block activates itself for a given amount of processed sequences
            self.sequences_processed += 1
            # if self.sequences_processed % self.activeation_seq_len != 0:
            if self.sequences_processed < self.activation_seq_len:
                self.output_flag_buffer.put(True) # Do not add delay to self-flagging blocks
            else:
                self.sequences_processed -= self.activation_seq_len # Prevent overflow


    def describe(self):
        """Describe the block (verbose-like).

        :return: Dictionary comprising the block's description.
        """
        description = super().describe()

        description['activation_seq_len'] = self.activation_seq_len
        description['input_flag_buffer_block_id'] = id(self.input_flag_buffer)
        description['output_flag_buffer_block_id'] = id(self.output_flag_buffer)

        return description



class JointCFOAndIQImbalanceEstimation(IOFlaggedBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Joint CFO and IQ imbalance estimation', has_input_flag_buffer=True, **kwargs)

class NoiseAndChannelEstimation(IOFlaggedBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Noise and channel estimation', has_input_flag_buffer=False, **kwargs)

class JointIQImbalanceAndCFOCompensation(LoopbackFlaggedBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Joint IQ imbalance and CFO compensation', **kwargs)

class FrequencyDomainChannelEqualization(LoopbackFlaggedBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Frequency domain channel equalization', **kwargs)



class DetectionAndCoarseEstimation(IOFlaggedBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Detection and coarse estimation', **kwargs)

class IQImbalanceEstimation(IOFlaggedBlock):
    def __init__(self, **kwargs):
        super().__init__(name='IQ imbalance estimation', has_input_flag_buffer=True, **kwargs)

class Finefine_CFO_estimation_blockstimation(IOFlaggedBlock):
    def __init__(self, **kwargs):
        super().__init__(name='CFO estimation', **kwargs)

class ChannelEstimation(IOFlaggedBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Channel estimation', **kwargs)

class NoiseEstimation(IOFlaggedBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Noise estimation', **kwargs)

class IQImbalanceCompensation(LoopbackFlaggedBlock):
    def __init__(self, **kwargs):
        super().__init__(name='IQ imbalance compensation', **kwargs)

class CFOCompensation(LoopbackFlaggedBlock):
    def __init__(self, **kwargs):
        super().__init__(name='CFO compensation', **kwargs)

class FFT(FlaglessBlock):
    def __init__(self, **kwargs):
        super().__init__(name='FFT', **kwargs)

class ChannelEqualisation(LoopbackFlaggedBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Channel Equalisation', **kwargs)

class IFFT(FlaglessBlock):
    def __init__(self, **kwargs):
        super().__init__(name='IFFT', **kwargs)

class PilotBasedTracking(FlaglessBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Pilot-based tracking', **kwargs)

class PhaseCorrection(FlaglessBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Phase correction', **kwargs)

# class Demapper(LoopbackFlaggedBlock):
class Demapper(FlaglessBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Demapper', **kwargs)

class Decoder(FlaglessBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Decoder', **kwargs)

class Descrambler(FlaglessBlock):
    def __init__(self, **kwargs):
        super().__init__(name='Descrambler', **kwargs)


### Aggregation / segregation block definitions ########################################################################

class AggregationBlankBlock(BufferedBlock):
    """Block used for connecting a component's output to a data buffer with larger unit sizes.

    Accumulates the data in a 'tampon' buffer and forwards it once a certain threshold is reached.
    Unlike normal blocks, it does not induce any additional delays.
    """

    def __init__(self, agg_len, **kwargs):
        """
        :param agg_len: Threshold after which the data is forwarded to the output buffer.
        :param kwargs: Keyword arguments
        """

        # super().__init__(name, **kwargs)
        super().__init__(**kwargs)
        self.agg_seq_len = agg_len
        self.agg_buffer = np.empty((0))

    def run(self):
        while True:

            sequence = yield self.input_data_buffer.get()

            # The data gets accumulated in an internal buffer till the threshold is surpassed
            self.agg_buffer = np.concatenate((self.agg_buffer, sequence), axis=0)
            num_of_agg_seq = int(self.agg_buffer.size / self.agg_seq_len)
            for i in range(num_of_agg_seq):  # If at least one entire sequence is present in the aggregation buffer
                self.output_data_buffer.put( self.agg_buffer[:self.agg_seq_len] )  # Add sequence to output buffer
                self.agg_buffer = np.delete(self.agg_buffer, [*range(self.agg_seq_len)])  # Empty buffer



class SegregationBlankBlock(BufferedBlock):
    """Block used for connecting a component's output to a data buffer with smaller unit sizes.

    Spreads the data into smaller units and forwards it. For example 448 data blocks into indivblock_idual symbols.
    Unlike normal blocks, it does not induce any additional delays.
    """

    def __init__(self, seq_len=1, **kwargs):
        """
        :param kwargs: Keyword arguments
        :type kwargs: list
        """

        super().__init__(**kwargs)
        self.seq_len = seq_len

    def run(self):
        while True:

            sequence = yield self.input_data_buffer.get()

            seq_count = sequence.shape[0] / self.seq_len

            if seq_count % 1 != 0:
                raise ValueError(f'Non-ideal segregation from shape {sequence.shape[0]} to {self.seq_len}')

            for i in range(int(seq_count)):
                self.output_data_buffer.put( sequence[i*self.seq_len : (i+1)*self.seq_len] )


### Simplified block definitions #######################################################################################

class CEST_Simple(BufferedBlock):
    """Channel estimation block.

    Raises a positive flag for the FDE once the propagation delay has elapsed. Has no further function.
    """

    def __init__(self, block_id, env, log_path, log_subdir, delay):
        super().__init__(block_id, 'Joint IQ imbalance and CFO compensation', env, log_path, log_subdir)
        self.delay = delay
        self.output_flag_buffer = None

    def run(self):

        while True:

            yield self.input_data_buffer.get() # Wait till the input sequence arrives
            # With a single expected input sequence, only the propagation delay is used
            delay = self.delay.get_propagation_delay()
            yield self.env.timeout(delay)
            # Enable channel equalization
            self.output_flag_buffer.put(True)


class FDE_Simple(BufferedBlock):
    """Self-flagged frequency domain equalization block.

    Includes hardcoded properties - designed bearing in mind the succeeding block is the demapper.
    """

    def __init__(self, block_id, env, log_path, log_subdir, delay, func):

        name = 'Frequency domain channel equalization'

        super().__init__(block_id, name, env, log_path, log_subdir)

        self.delay = delay
        self.func = func

        log_name = f'{block_id:02}-{name}-input_flag_buffer_log'
        self.input_flag_buffer = Buffer(env, log_name, log_path, log_subdir)
        self.add_log(self.input_flag_buffer.log)

        self.output_flag_buffer = self.input_flag_buffer

    def run(self):

        while True:

            # Flagged blocks can only run once they receive confirmation.
            if not self.input_flag_buffer is None:
                yield self.input_flag_buffer.get()  # Wait for flag (green light)
                self.input_flag_buffer.log_get_granted()  # Only log once the resource has been successfully fetched

            # Include TD
            delay = self.delay.get_throughput_delay()
            yield self.env.timeout(delay)

            # The block waits for data to become available before resuming
            sequence = yield self.input_data_buffer.get()
            self.input_data_buffer.log_get_granted()  # Add log entry once the resource has been successfully fetched

            # The block modifies the sequence
            sequence = self.func.run(sequence)

            # Move to output and include PD
            self.env.process(self.move_to_output_buffer(sequence))

            # The block re-activates itself for a given amount of processed sequences
            self.output_flag_buffer.put(True) # Do not add additional delay to self-flagging blocks


    def move_to_output_buffer(self, sequence):
        """Move sequence to demapper input buffer and add propagation delay.
        :param sequence: Equalized block without GI.
        """

        delay = self.delay.get_propagation_delay()
        yield self.env.timeout(delay)

        # Forward the symbols to the demapper
        for symbol in sequence:
            self.output_data_buffer.put(symbol)


    def describe(self):
        """Describe the block (verbose-like).

        :return: Dictionary comprising the block's description.
        """
        description = super().describe()

        description['input_flag_buffer_block_id'] = id(self.input_flag_buffer)

        return description


class Demapper_Simple(BufferedBlock):
    """Demapper instance.

    Apart from spawning fictitious LLRs from fictitious symbols, doesn't include any further functionality.
    Includes hardcoded properties - designed bearing in mind the succeeding block is the decoder.
    """

    def __init__(self, block_id, env, log_path, log_subdir, delay, func):

        super().__init__(block_id, 'Demapper', env, log_path, log_subdir)

        self.delay = delay
        self.func = func

        self.reset_cw_buffer()

    def run(self):

        while True:

            # Include TD
            delay = self.delay.get_throughput_delay()
            yield self.env.timeout(delay)

            # The block waits for data to become available before resuming
            symbol = yield self.input_data_buffer.get()
            self.input_data_buffer.log_get_granted()  # Add log entry once the resource has been successfully fetched

            # The block modifies the sequence
            llr = self.func.run(symbol)

            # Move to output and include PD
            self.env.process(self.move_to_output_buffer(llr))

    def move_to_output_buffer(self, llr):
        """Move sequence to decoder input buffer and add propagation delay.
        :param sequence: llr sequence.
        """

        # Wait for propagation delay associated with each symbol (llr sequence)
        delay = self.delay.get_propagation_delay()
        yield self.env.timeout(delay)

        self.add_to_cw_buffer(llr)

        # When an entire codeword has aggregated, forward it to the decoder
        if self.is_cw_buffer_full():
            self.output_data_buffer.put(self.cw_buffer)  # Immediately forward input sequence to output buffer
            self.reset_cw_buffer()

    def reset_cw_buffer(self):
        """Empty out cw buffer and reset write counter"""
        self.cw_buffer = np.zeros(672)
        self.cw_buffer_write_idx = 0

    def add_to_cw_buffer(self, llr):
        """Add llr sequence to cw buffer"""
        size = llr.size
        self.cw_buffer[ self.cw_buffer_write_idx : self.cw_buffer_write_idx + size ] = llr
        self.cw_buffer_write_idx += size

    def is_cw_buffer_full(self):
        """Check if cw buffer is full"""
        if self.cw_buffer_write_idx < 672:
            return False
        elif self.cw_buffer_write_idx == 672:
            return True
        else:
            raise RuntimeError('Codeword buffer is not multiple of codeword size')


class Decoder_Simple(BufferedBlock):
    """Decoder instance.

    Apart from reducing the sequence size from CW to DW, doesn't include any further functionality.
    Includes hardcoded properties - designed bearing in mind the succeeding block is the decoder.
    """

    def __init__(self, block_id, env, log_path, log_subdir, delay, func):

        super().__init__(block_id, 'Decoder', env, log_path, log_subdir)

        self.delay = delay
        self.func = func

        # Output data buffer is also the end of the line
        log_name = f'{block_id+1:02}-output'
        self.output_data_buffer = Buffer( env, log_name, log_path, log_subdir)
        self.add_log(self.output_data_buffer.log)

        self.processed_bit_counter = 0 # For monitoring purposes, track the number of processed bits

        self.first_run = True # Flag for skipping TD on first run

    def run(self):

        while True:


            if not self.first_run:
                delay = self.delay.get_throughput_delay() # Include TD
                delay = delay * 672 * self.func.code_rate # Multipli the Average TD by the number of output data bits
                yield self.env.timeout(delay)
            else:
                self.first_run = False # On next yield take delay into account

            # The block waits for data to become available before resuming
            codeword = yield self.input_data_buffer.get()
            self.input_data_buffer.log_get_granted()  # Add log entry once the resource has been successfully fetched

            # The block modifies the sequence
            dataword = self.func.run(codeword)

            # Move to output and include PD
            self.env.process(self.move_to_output_buffer(dataword))

            # For monitoring purposes, output the number of processed bits
            self.processed_bit_counter += dataword.size
            print( f'{dt.now()} | Processed a total of {self.processed_bit_counter} bits' )
            sys.stdout.flush()

    def move_to_output_buffer(self, dataword):
        """Move sequence to output and add propagation delay.
        :param sequence: llr sequence.
        """

        # Wait for propagation delay associated with each symbol (llr sequence)
        delay = self.delay.get_propagation_delay()
        yield self.env.timeout(delay)

        self.output_data_buffer.put(dataword)


class FDE_Simple448(BufferedBlock):
    """Self-flagged frequency domain equalization block.

    Includes hardcoded properties - designed bearing in mind the succeeding block is the demapper.
    Forwards entire 448-long sequences to the demapper to reduce number of events.
    """

    def __init__(self, block_id, env, log_path, log_subdir, delay, func):

        name = 'Frequency domain channel equalization'

        super().__init__(block_id, name, env, log_path, log_subdir)

        self.delay = delay
        self.func = func

        log_name = f'{block_id:02}-{name}-input_flag_buffer_log'
        self.input_flag_buffer = Buffer(env, log_name, log_path, log_subdir)
        self.add_log(self.input_flag_buffer.log)

        self.output_flag_buffer = self.input_flag_buffer

    def run(self):

        while True:

            # Flagged blocks can only run once they receive confirmation.
            if not self.input_flag_buffer is None:
                yield self.input_flag_buffer.get()  # Wait for flag (green light)
                self.input_flag_buffer.log_get_granted()  # Only log once the resource has been successfully fetched

            # Include TD
            delay = self.delay.get_throughput_delay()
            yield self.env.timeout(delay)

            # The block waits for data to become available before resuming
            sequence = yield self.input_data_buffer.get()
            self.input_data_buffer.log_get_granted()  # Add log entry once the resource has been successfully fetched

            # The block modifies the sequence
            sequence = self.func.run(sequence)

            # Move to output and include PD
            self.env.process(self.move_to_output_buffer(sequence))

            # The block re-activates itself for a given amount of processed sequences
            self.output_flag_buffer.put(True) # Do not add additional delay to self-flagging blocks


    def move_to_output_buffer(self, sequence):
        """Move sequence to demapper input buffer and add propagation delay.
        :param sequence: Equalized block without GI.
        """

        delay = self.delay.get_propagation_delay()
        yield self.env.timeout(delay)

        # The demapper works with 448-sequences to reduce the amount of events
        self.output_data_buffer.put(sequence)


    def describe(self):
        """Describe the block (verbose-like).

        :return: Dictionary comprising the block's description.
        """
        description = super().describe()

        description['input_flag_buffer_block_id'] = id(self.input_flag_buffer)

        return description


class Demapper_Simple448(BufferedBlock):
    """Demapper instance.

    Apart from spawning fictitious llrs from fictitious symbols, doesn't include any further functionality.
    Includes hardcoded properties - designed bearing in mind the succeeding block is the decoder.
    Works with entire 448-long input sequences to reduce the number of events.
    """

    def __init__(self, block_id, env, log_path, log_subdir, delay, func):

        super().__init__(block_id, 'Demapper', env, log_path, log_subdir)

        self.delay = delay
        self.func = func

        self.symbols_in_buffer = 0

        self.symbols_per_cw = int(672 / self.func.modulation_rate)


    def run(self):

        while True:

            # The block waits for data to become available before resuming
            # yield self.input_data_buffer.get()
            sequence = yield self.input_data_buffer.get()
            self.input_data_buffer.log_get_granted()  # Add log entry once the resource has been successfully fetched

            # self.symbols_in_buffer += 448
            self.symbols_in_buffer += sequence.size

            # Mimic sequential processing of symbols before forwarding them to the decoder
            while self.symbols_in_buffer >= self.symbols_per_cw:
                delay = self.delay.get_throughput_delay() * self.symbols_per_cw # On the first run, this unadvertedly adds one symbol TD
                yield self.env.timeout(delay)
                self.symbols_in_buffer -= self.symbols_per_cw
                self.output_data_buffer.put(np.zeros(672)) # Change this to only pass single value


    def move_to_output_buffer(self, llr):
        """Move sequence to decoder input buffer and add propagation delay.
        :param sequence: llr sequence.
        """

        # Wait for propagation delay associated with each symbol (llr sequence)
        delay = self.delay.get_propagation_delay()
        yield self.env.timeout(delay)

        self.add_to_cw_buffer(llr)

        # When an entire codeword has aggregated, forward it to the decoder
        # if self.is_cw_full_available():
        while self.is_cw_full_available():
            self.output_data_buffer.put(self.cw_buffer)  # Immediately forward input sequence to output buffer
            # self.reset_cw_buffer()
            self.cw_buffer_write_idx -= 672

    def reset_cw_buffer(self):
        """Empty out cw buffer and reset write counter"""
        self.cw_buffer = np.zeros(672)
        self.cw_buffer_write_idx = 0

    def add_to_cw_buffer(self, llr):
        """Add llr sequence to cw buffer"""
        size = llr.size
        # self.cw_buffer[ self.cw_buffer_write_idx : self.cw_buffer_write_idx + size ] = llr
        self.cw_buffer_write_idx += size

    def is_cw_buffer_full(self):
        """Check if cw buffer is full"""
        if self.cw_buffer_write_idx < 672:
            return False
        elif self.cw_buffer_write_idx == 672:
            return True
        else:
            raise RuntimeError('Codeword buffer is not multiple of codeword size')

    def is_cw_full_available(self):
        """Check if full cw is available"""
        return self.cw_buffer_write_idx >= 672


# class JCOMP_Simple(BufferedBlock):
#     """Joint IQ imbalance and CFO compensation.
#
#     Aggregates incoming symbols into blocks and forwards them to the FDE.
#     """
#
#     def __init__(self, block_id, env, log_path, log_subdir):
#         super().__init__(block_id, 'Joint IQ imbalance and CFO compensation', env, log_path, log_subdir)
#         self.reset_block_buffer()
#
#     def run(self):
#
#         while True:
#
#             symbol = yield self.input_data_buffer.get() # Get input symbol when available
#             self.add_to_block_buffer(symbol)
#
#             # When an entire block has aggregated, forward it to the FDE
#             if self.is_block_buffer_full():
#                 self.output_data_buffer.put(self.block_buffer) # Immediately forward input sequence to output buffer
#                 self.reset_block_buffer()
#
#     def reset_block_buffer(self):
#         self.block_buffer = np.zeros(512, dtype=complex)
#         self.block_buffer_write_idx = 0
#
#     def add_to_block_buffer(self, symbol):
#         self.block_buffer[ self.block_buffer_write_idx ] = symbol
#         self.block_buffer_write_idx += 1
#
#     def is_block_buffer_full(self):
#         return self.block_buffer_write_idx