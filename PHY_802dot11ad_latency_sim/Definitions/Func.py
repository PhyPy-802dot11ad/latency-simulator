import numpy as np

# import phy802dot11ad

# import Definitions.CustomDecoder as CDEC
# import Definitions.CustomDemapper as CDEM


class Functionality():
    """Class representing the functionality of a component block."""

    def __init__(self, description=''):
        """
        :param description: Short description of the functionality.
        :type description: str
        """
        self.description = description

    def describe(self):
        pass


class Blank(Functionality):

    def __init__(self, **kwargs):
        description = 'Blank placeholder'
        super().__init__(description, **kwargs)

    def run(self, sequence):
        # return sequence, self.static_delay
        return sequence

    def describe(self):
        return None


class Demapper(Functionality):

    def __init__(self, modulation_rate, **kwargs):
        """
        :param modulation_rate:
        :param noise_variance:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.modulation_rate = modulation_rate

    def run(self, sequence):
        """Demap symbol or sequence of symbols.

        :param sequence: Single or multiple mapped symbols.
        """
        demapped_sequence = np.zeros(sequence.size*self.modulation_rate, dtype=np.uint8)
        return demapped_sequence

    def describe(self):
        return None


class Decoder(Functionality):

    def __init__(self, code_rate, num_of_cw, cw_padding, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.code_rate = code_rate
        self.codeword_length = int(672*code_rate)
        self.cw_to_process = num_of_cw
        self.cw_padding = cw_padding

    def run(self, codeword):
        """Decode codeword.
        """
        dataword = codeword[:self.codeword_length] # Remove parity bits
        self.cw_to_process -= 1
        if self.cw_to_process == 0:
            if self.cw_padding > 0:
                dataword = dataword[:-self.cw_padding] # Remove padding bits
        return dataword

    def describe(self):
        return None


class FDE(Functionality):

    def __init__(self, number_of_blocks, number_of_block_padding_bits, **kwargs):
        """
        :param number_of_blocks: Blocks per PPDU.
        :param number_of_block_padding_bits: Padding bits at the end of the final block.
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.blocks_to_process = number_of_blocks
        self.number_of_block_padding_bits = number_of_block_padding_bits

    def run(self, sequence):
        """Cut away the GI at the end of the block, and cut off the padding bits at the end of the last block.

        :param sequence: Block
        """

        sequence = sequence[64:] # Remove pre-pended GI

        # Remove block padding bits (last block)
        self.blocks_to_process -= 1
        if self.blocks_to_process == 0:
            if self.number_of_block_padding_bits > 0:
                sequence = sequence[:-self.number_of_block_padding_bits]

        return sequence

    def describe(self):
        return None
