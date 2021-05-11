"""


Definitions:
    - propagation_delay: Latency
    - throughput_delay: Inverse throughput

"""


import warnings


class Delay():
    """Class representing the delay of a component block."""

    # def __init__(self, source_description='', throughput_delay=0):
    def __init__(self, throughput_delay, propagation_delay, source_description='' ):
        """
        :param source_description: Short source_description of the delay and its origin in literature.
        :type source_description: str
        :param throughput_delay: Fixed delay, not dependent on the sequence length.
        :type throughput_delay: float or int
        """

        self.source_description = source_description
        # self.throughput_delay = throughput_delay
        self.propagation_delay = propagation_delay
        self.throughput_delay = throughput_delay


    def get_throughput_delay(self):
        """Expose the delay incurred due to finite throughput.

        :return: Delay.
        :rtype: float or int
        """

        return self.throughput_delay

    def get_propagation_delay(self):
        """Expose the delay incurred due to the propagation of data within the component.

        :return: Delay.
        :rtype: float or int
        """

        return self.propagation_delay


    def describe(self):
        description = {
            'source_description': self.source_description,
            'throughput_delay': self.get_throughput_delay(),
            'propagation_delay': self.get_propagation_delay()
        }
        return description


class DecoderWiener28nm(Delay):
    """
    Decoder delay metrics based on the 28 nm LDPC decoder:
    - Weiner et. al (2016), A Scalable 1.5-to-6Gb/s 6.2-to-38.1mW LDPC Decoder for 60GHz Wireless Networks in 28nm UTBB FDSOI
    """

    def __init__(self, code_rate, **kwargs):
        """
        :param code_rate: Code rate
        :type code_rate: float
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """

        source_description = "Decoder delay metrics based on the 28 nm LDPC decoder: Weiner et. al (2016), A Scalable..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(code_rate)

    def set_throughput_delay(self, code_rate):
        """Set the throughput delay.

        :param code_rate: Code rate
        :type code_rate: float
        """

        if code_rate == 1/2:
            self.throughput_delay = 224
        elif code_rate == 5/8:
            self.throughput_delay = 140
        elif code_rate == 3/4:
            self.throughput_delay = 84
        elif code_rate == 13/16:
            self.throughput_delay = 46
        else:
            self.throughput_delay = 100
            warnings.warn(f"Defaulting to a delay of {self.throughput_delay} ns for code rate {code_rate}", RuntimeWarning)


class DecoderWiener65nm(Delay):
    """
    Decoder delay metrics based on the 28 to 65 nm transformation of Weiner's LDPC decoder, found in (corresponding to reference 75):
    - Shao et. al (2019), Survey of Turbo, LDPC, and Polar Decoder ASIC Implementations
    """

    def __init__(self, code_rate, **kwargs):
        """
        :param code_rate: Code rate.
        :type code_rate: float
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """

        source_description = "Decoder delay metrics based on the 28 to 65 nm transformation of Weiner's LDPC decoder, found in (ref. 75): Shao et. al (2019), Survey of Turbo, LDPC..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(code_rate)

    def set_throughput_delay(self, code_rate):
        """Set the throughput delay.

        :param code_rate: Code rate.
        :type code_rate: float
        """

        if code_rate == 1/2:
            self.throughput_delay = 520
        elif code_rate == 5/8:
            self.throughput_delay = 325
        elif code_rate == 3/4:
            self.throughput_delay = 195
        else:
            self.throughput_delay = 100
            warnings.warn(f"Defaulting to a delay of {self.throughput_delay} ns for code rate {code_rate}", RuntimeWarning)


class DecoderLi_1_2(Delay):
    """
    Decoder delay metrics based on the 28 nm LDPC decoder:
    - Li et. al (2015), Area and energy efficient 802.11ad LDPC decoding processor

    The decoder only supports 1/2 code rate. The figures apply to 3-iteration decoding.
    """

    def __init__(self, code_rate, **kwargs):
        """
        :param code_rate: Code rate.
        :type code_rate: float
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """

        source_description = "Decoder delay metrics based on the 28 nm LDPC decoder: Li et. al (2015), Area and energy efficient 802.11ad LDPC decoding processor"
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(code_rate)
        self.set_propagation_delay(code_rate)

    def set_throughput_delay(self, code_rate):
        """Set the throughput delay.

        :param code_rate: Code rate.
        :type code_rate: float
        """

        self.throughput_delay = 0.14 * 672 * 1 / 2 # Inverse throughput per dataword
        if code_rate != 1 / 2:
            warnings.warn(f"Defaulting to a delay of {self.throughput_delay} ns for code rate {code_rate}", RuntimeWarning)

    def set_propagation_delay(self, code_rate):
        """Set the propagation delay.

        :param code_rate: Code rate.
        :type code_rate: float
        """

        self.propagation_delay = 95
        if code_rate != 1 / 2:
            warnings.warn(f"Defaulting to a delay of {self.propagation_delay} ns for code rate {code_rate}", RuntimeWarning)


class DecoderLi_13_16(Delay):
    """
    Decoder delay metrics based on the 28 nm LDPC decoder:
    - Li et. al (2013), A processor based multi-standard low-power LDPC engine for multi-Gbps wireless communication

    The decoder only supports 13/16 code rate. The figures apply to 5-iteration decoding.
    """

    def __init__(self, code_rate, **kwargs):
        """
        :param code_rate: Code rate.
        :type code_rate: float
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """

        source_description = "Decoder delay metrics based on the 28 nm LDPC decoder: Li et. al (2013), A processor based multi-standard low-power LDPC..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(code_rate)
        self.set_propagation_delay(code_rate)

    def set_throughput_delay(self, code_rate):
        """Set the throughput delay.

        :param code_rate: Code rate.
        :type code_rate: float
        """

        self.throughput_delay = 0.19 * 672 * 13 / 16 # Inverse throughput per dataword
        if code_rate != 13 / 16:
            warnings.warn(f"Defaulting to a delay of {self.throughput_delay} ns for code rate {code_rate}", RuntimeWarning)

    def set_propagation_delay(self, code_rate):
        """Set the propagation delay.

        :param code_rate: Code rate.
        :type code_rate: float
        """

        self.propagation_delay = 150
        if code_rate != 13 / 16:
            warnings.warn(f"Defaulting to a delay of {self.propagation_delay} ns for code rate {code_rate}", RuntimeWarning)


class DecoderLiCustom(Delay):
    """
    Decoder delay metrics based on the 28 nm LDPC decoder:
    - Li et. al (2013), A processor based multi-standard low-power LDPC engine for multi-Gbps wireless communication

    The performance figures were expanded based on the findings in the original article and several assumptions.
    View low-latency paper for more info.
    """

    REF_CYCLES = 13
    REF_ITERATIONS = 5
    # CYCLES_PER_ITERATION = [20.67, 19.67, 17, 13]

    def __init__(self, code_rate, iterations, **kwargs):
        """
        :param code_rate: Code rate.
        :type code_rate: float
        :param iterations: Code rate.
        :type iterations: int
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """

        source_description = "Decoder delay metrics based on the 28 nm LDPC decoder: Li et. al (2013), A processor based multi-standard low-power LDPC..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(code_rate, iterations)
        self.set_propagation_delay(code_rate, iterations)

    def set_throughput_delay(self, code_rate, iterations):
        """Set throughput delay relative to the code rate 13/16 and 5 iterations"""
        cycles = self.__get_number_of_cycles(code_rate)
        # self.throughput_delay = (5.3 * 10**9 * cycles/self.REF_CYCLES)**(-1) * iterations/self.REF_ITERATIONS
        # self.throughput_delay = (5.3 * cycles/self.REF_CYCLES)**(-1) * iterations/self.REF_ITERATIONS
        self.throughput_delay = (5.3 * self.REF_CYCLES/cycles)**(-1) * iterations/self.REF_ITERATIONS

    def set_propagation_delay(self, code_rate, iterations):
        """Set propagation delay relative to the code rate 13/16 and 5 iterations"""
        cycles = self.__get_number_of_cycles(code_rate)
        self.propagation_delay = 150 * (0.25 + 0.75 * cycles/self.REF_CYCLES * iterations/self.REF_ITERATIONS )

    def __get_number_of_cycles(self, code_rate):
        if code_rate == 1/2: return 20.67
        elif code_rate == 5/8: return 19.67
        elif code_rate == 3/4: return 17
        elif code_rate == 13/16: return 13


class DemapperJafriSuboptimalConventionalFPGA65nm(Delay):
    """
    Demapper delay for conventional suboptimal soft demapping using an FPGA in 65 nm technology:
    - Jafri et. al (2010), Rapid Design and Prototyping of Universal Soft Demapper
    """

    def __init__(self, modulation_rate, **kwargs):
        """
        :param code_rate: Modulation rate.
        :type code_rate: float
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """

        source_description = "Demapper delay for conventional suboptimal soft demapping using an FPGA in 65 nm technology: Jafri et. al (2010), Rapid Design..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(modulation_rate)

    def set_throughput_delay(self, modulation_rate):
        """Set the throughput delay.

        :param code_rate: Modulation rate.
        :type code_rate: float
        """

        # self.throughput_delay = 38  # 26 Msym/s for 16QAM
        self.throughput_delay = 38
        if modulation_rate != 4:
            warnings.warn(f"Unsupported modulatino rate {modulation_rate}, defaulting to a delay of {self.throughput_delay} ns per symbol (16QAM)", RuntimeWarning)


class DemapperJafriSuboptimalConventionalFPGAtoASIC65nm(Delay):
    """
    Demapper delay for conventional suboptimal soft demapping using an 65 nm FPGA, transformed to 65 nm ASIC using 3.2 speed factor:
    - Kuon (2006) Measuring the Gap between FPGAs and ASICs
    - Jafri et. al (2010), Rapid Design and Prototyping of Universal Soft Demapper
    """

    def __init__(self, modulation_rate, **kwargs):
        """
        :param code_rate: Modulation rate.
        :type code_rate: float
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """

        source_description = "Demapper delay for conventional suboptimal soft demapping using an 65 nm FPGA, transformed to 65 nm ASIC using 3.2 speed factor: Kuon (2006) Measuring the Gap... Jafri et. al (2010), Rapid Design..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(modulation_rate)

    def set_throughput_delay(self, modulation_rate):
        """Set the throughput delay.

        :param code_rate: Modulation rate.
        :type code_rate: float
        """

        self.throughput_delay = 12  # 26*3.2 Msym/s for 16QAM
        if modulation_rate != 4:
            warnings.warn(f"Unsupported modulatino rate {modulation_rate}, defaulting to a delay of {self.throughput_delay} ns per symbol (16QAM)", RuntimeWarning)

class DemapperJafriOptimal(Delay):
    """
    Demapper delay for optimal soft demapping using an 28 nm FPGA, transformed to 28 nm ASIC using 3.2 speed factor:
    - Kuon (2006) Measuring the Gap between FPGAs and ASICs
    - Jafri et. al (2017), High-Throughput and Area-Efficient Rotated and Cyclic Q Delayed Constellations Demapper for Future Wireless Standards
    """

    def __init__(self, modulation_rate, **kwargs):
        """
        :param code_rate: Modulation rate.
        :type code_rate: float
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """

        source_description = "Demapper delay for optimal soft demapping using an 28 nm FPGA, transformed to 28 nm ASIC using 3.2 speed factor: Kuon (2006) Measuring the Gap... Jafri et. al (2017), High-Throughput..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(modulation_rate)

    def set_throughput_delay(self, modulation_rate):
        """Set the throughput delay.

        :param code_rate: Modulation rate.
        :type code_rate: float
        """

        if modulation_rate == 1: self.throughput_delay = 0.83
        elif modulation_rate == 2: self.throughput_delay = 1.67
        elif modulation_rate == 4: self.throughput_delay = 5
        elif modulation_rate == 6: self.throughput_delay = 18.29


class DemapperJafriSuboptimal(Delay):
    """
    Demapper delay for suboptimal soft demapping using an ? nm ASIC:
    - Jafri et. al (2009), ASIP-Based Universal Demapper for Multiwireless Standards
    """

    def __init__(self, modulation_rate, **kwargs):
        """
        :param code_rate: Modulation rate.
        :type code_rate: float
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """

        source_description = "Demapper delay for optimal soft demapping using an 28 nm FPGA, transformed to 28 nm ASIC using 3.2 speed factor: Kuon (2006) Measuring the Gap... Jafri et. al (2017), High-Throughput..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(modulation_rate)

    def set_throughput_delay(self, modulation_rate):
        """Set the throughput delay.

        :param code_rate: Modulation rate.
        :type code_rate: float
        """

        if modulation_rate == 1: self.throughput_delay = 0.44
        elif modulation_rate == 2: self.throughput_delay = 0.88
        elif modulation_rate == 4: self.throughput_delay = 1.32
        elif modulation_rate == 6: self.throughput_delay = 2.2


class DemapperAliDecisionThreshold65nm(Delay):
    """
    Demapper delay for decision threshold soft demapping algorithm, based on 65 nm technology:
    - Ali et. al (2015), A high throughput architecture for a low complexity soft-output demapping algorithm
    """

    def __init__(self, modulation_rate, **kwargs):
        """
        :param code_rate: Modulation rate.
        :type code_rate: float
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """

        source_description = "Demapper delay for decision threshold soft demapping algorithm, based on 65 nm technology: Ali et. al (2015), A high throughput arch..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(modulation_rate)

    def set_throughput_delay(self, modulation_rate):
        """Set the throughput delay.

        :param code_rate: Modulation rate.
        :type code_rate: float
        """

        if modulation_rate == 1: self.throughput_delay = 0.16
        elif modulation_rate == 2: self.throughput_delay = 0.31
        elif modulation_rate == 4: self.throughput_delay = 0.62
        elif modulation_rate == 6: self.throughput_delay = 0.93


class DemapperCustomOptimal(Delay):
    """
    Throughput reported in (1) and further multiplied by 3.2, based on (2):
    - Jafri et. al (2017), High-Throughput and Area-Efficient Rotated and Cyclic Q Delayed Constellations Demapper for Future Wireless Standards
    - Kuon (2006) Measuring the Gap between FPGAs and ASICs
    """

    def __init__(self, modulation_rate, **kwargs):
        """
        :param code_rate: Modulation rate
        """

        source_description = "Optimal demapper delay: Jafri et. al (2017), High-Throughput and Area-Efficient..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(modulation_rate)

    def set_throughput_delay(self, modulation_rate):
        """Set the throughput delay."""

        if modulation_rate == 1:
            Warning('BPSK wasn\'t covered in the source paper, setting demapper delay to QPSK.')
            self.set_throughput_delay(2)
        elif modulation_rate == 2: self.throughput_delay = 1.67  # 75 * 3.2 = 240 MLLR/s     ->  1200 MLLR/s (5 parallel) -> 600 Msys/s
        elif modulation_rate == 4: self.throughput_delay = 5.0   # 50 * 3.2 = 160 MLLR/s     ->   800 MLLR/s (5 parallel) -> 200 Msys/s
        elif modulation_rate == 6: self.throughput_delay = 18.29 # 20.5 * 3.2 = 65.6 MLLR/s  ->   328 MLLR/s (5 parallel) -> 54.7 Msys/s


class DemapperCustomOptimal448(DemapperCustomOptimal):

    def __init__(self, modulation_rate, **kwargs):
        super().__init__(modulation_rate, **kwargs)
        self.throughput_delay *= 448


class DemapperCustomSubOptimal(Delay):
    """
    Throughput reported in:
    - Jafri et. al (2009), ASIP-Based Universal Demapper for Multiwireless Standards
    """

    def __init__(self, modulation_rate, **kwargs):
        source_description = "Sub-Optimal demapper delay: Jafri et. al (2009), ASIP-Based Universal Demapper for..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(modulation_rate)

    def set_throughput_delay(self, modulation_rate):
        """Set the throughput delay."""

        if modulation_rate == 1:
            Warning('BPSK wasn\'t covered in the source paper, setting demapper delay to QPSK.')
            self.set_throughput_delay(2)
        elif modulation_rate == 2: self.throughput_delay = 0.88 # 454 MLLR/s  ->  2270 MLLR/s (5 parallel) -> 1135 Msys/s
        elif modulation_rate == 4: self.throughput_delay = 1.32 # 606 MLLR/s  ->  3030 MLLR/s (5 parallel) -> 757,5 Msys/s
        elif modulation_rate == 6: self.throughput_delay = 2.2  # 545 MLLR/s  ->  2725 MLLR/s (5 parallel) -> 454.17 Msys/s


class DemapperCustomSubOptimal448(DemapperCustomSubOptimal):

    def __init__(self, modulation_rate, **kwargs):
        super().__init__(modulation_rate, **kwargs)
        self.throughput_delay *= 448


class DemapperCustomDecisionThreshold(Delay):
    """
    Throughput reported in:
    - Ali et. al (2015), A high throughput architecture for a low complexity soft-output demapping algorithm
    """

    def __init__(self, modulation_rate, **kwargs):
        source_description = "Decision threshold demapper delay: Ali et. al (2015), A high throughput architecture for"
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay(modulation_rate)

    def set_throughput_delay(self, modulation_rate):
        """Set the throughput delay."""

        if modulation_rate != 4:
            Warning('Selected modulation rate wasn\'t benchmarked in the source study, using 16QAM delay instead.')

        self.throughput_delay = 0.6  # 1328 MLLR/s  ->  6640 MLLR/s (5 parallel) -> 1660 Msys/s

        # if modulation_rate == 1:   self.throughput_delay = 0.15 # 1328 MLLR/s  ->  6640 MLLR/s (5 parallel) -> 6640 Msys/s
        # elif modulation_rate == 2: self.throughput_delay = 0.3  # 1328 MLLR/s  ->  6640 MLLR/s (5 parallel) -> 3320 Msys/s
        # elif modulation_rate == 4: self.throughput_delay = 0.6  # 1328 MLLR/s  ->  6640 MLLR/s (5 parallel) -> 1660 Msys/s
        # elif modulation_rate == 6: self.throughput_delay = 0.9  # 1328 MLLR/s  ->  6640 MLLR/s (5 parallel) -> 1106,67 Msys/s


class DemapperCustomDecisionThreshold448(DemapperCustomDecisionThreshold):

    def __init__(self, modulation_rate, **kwargs):
        super().__init__(modulation_rate, **kwargs)
        self.throughput_delay *= 448


class DescramblerChen180nm(Delay):
    """
    Descrambler delay, based on 65 nm technology:
    - Chen et. al (2010), Efficient High-Throughput Architectures for High-Speed Parallel Scramblers
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """
        source_description = "Descrambler delay, based on 65 nm technology: Chen et. al (2010), Efficient High-Throughput..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay()

    def set_throughput_delay(self):
        """Set the throughput delay.
        """

        self.throughput_delay = 0.04  # 25.6 Gbps


class FFTAhmed(Delay):
    """
    512-point FFT delay, based on 65 nm technology:
    - Ahmed et. al (2011), A 512-point 8-parallel pipelined feedforward FFT for WPAN
    """

    def __init__(self, clock_MHz=330, **kwargs):
        """
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """
        source_description = "512-point FFT delay, based on 65 nm technology: Ahmed et. al (2011), A 512-point 8-parallel..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay()
        self.set_propagation_delay(clock_MHz)

    def set_throughput_delay(self):
        """Set the throughput delay.
        """

        self.throughput_delay = 194  # 2.64 GSps, blocks of 512 samples

    def set_propagation_delay(self, clock_MHz):
        """Set the propagation delay.

        :param clock_MHZ: The clock frequency in MHz
        :type clock_MHZ: int or float
        """

        self.clock_MHz = clock_MHz
        self.propagation_delay = (10**3)/clock_MHz * 63 # clock-frequency-in-GHz * delay-in-clock-cycles


class FDEAhmed(Delay):
    """
    Delay of FDE block, comptrised of FFT + ch. eq. (mat mul) + IFFT

    The throughput delay is assumed the same as for FFTAhmed, while the popagation delay is double.
    """

    def __init__(self, clock_MHz=330, **kwargs):
        """
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """
        source_description = "FDE delay, mostly based on: Ahmed et. al (2011), A 512-point 8-parallel..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay()
        self.set_propagation_delay(clock_MHz)

    def set_throughput_delay(self):
        """Set the throughput delay.
        """

        self.throughput_delay = 194  # 2.64 GSps, blocks of 512 samples

    def set_propagation_delay(self, clock_MHz):
        """Set the propagation delay.

        :param clock_MHZ: The clock frequency in MHz
        :type clock_MHZ: int or float
        """

        self.clock_MHz = clock_MHz
        self.propagation_delay = (10**3)/clock_MHz * 63 # clock-frequency-in-GHz * delay-in-clock-cycles
        self.propagation_delay *= 2 # FFT and IFFT equally contribute to the delay


class FFTLiu(Delay): # Not the most representative (4096-point FFT)
    """
    4096-point FFT delay, based on 65 nm technology:
    - Liu et. al (2019), A High-Flexible Low-Latency Memory-Based FFT Processor for 4G, WLAN, and Future 5G
    """

    def __init__(self, clock_MHz=250, **kwargs):
        """
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """
        source_description = "4096-point FFT delay, based on 65 nm technology: Ahmed et. al (2011), A 512-point 8-parallel..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_throughput_delay()
        self.set_propagation_delay(clock_MHz)

    def set_throughput_delay(self):
        """Set the throughput delay.
        """

        self.throughput_delay = 527  # 972 MSps, blocks of 512 samples

    def set_propagation_delay(self, clock_MHz):
        """Set the propagation delay.

        :param clock_MHZ: The clock frequency in MHz
        :type clock_MHZ: int or float
        """

        self.clock_MHz = clock_MHz
        self.propagation_delay = (10**3)/clock_MHz * 226 # clock-frequency-in-GHz * delay-in-clock-cycles (closest was the source_description of size 480)


class EqualizerCustom(Delay):
    """Custom equalizer delay definition.

    Assumes all 'Gij*Xi' complex number multiplications are done in parallel, while the summation is done
    stepwise-parallel at the end - sum value pairs till only one is left, giving a log2(N) complexity.

    :param FLOAT_MULTIPLICATION_CYCLES: Clock cycles needed per float multiplication.
    :type FLOAT_MULTIPLICATION_CYCLES: int
    :param FLOAT_ADDITION_CYCLES: Clock cycles needed per float addition.
    :type FLOAT_ADDITION_CYCLES: int
    :param SUMMATION_CYCLES: Clock cycles needed per for individual multiplication result summation.
    :type SUMMATION_CYCLES: int
    """

    FLOAT_MULTIPLICATION_CYCLES = 4
    FLOAT_ADDITION_CYCLES = 1
    SUMMATION_CYCLES = 9 # log2(512)

    def __init__(self, clock_MHz=330, **kwargs):
        """
        :param kwargs: Keyword arguments.
        :type kwargs: list
        """
        source_description = "4096-point FFT delay, based on 65 nm technology: Ahmed et. al (2011), A 512-point 8-parallel..."
        super().__init__(0, 0, source_description, **kwargs)
        self.set_propagation_delay(clock_MHz)

    def set_propagation_delay(self, clock_MHz):
        """Set the propagation delay.

        :param clock_MHZ: The clock frequency in MHz
        :type clock_MHZ: int or float
        """

        self.clock_MHz = clock_MHz
        cycles = self.FLOAT_MULTIPLICATION_CYCLES + self.FLOAT_ADDITION_CYCLES + self.SUMMATION_CYCLES
        self.propagation_delay = (10**3)/clock_MHz * cycles # clock-frequency-in-GHz * delay-in-clock-cycles
