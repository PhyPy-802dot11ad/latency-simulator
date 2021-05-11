from .Definitions import Block, Func, Delay, Buffer

from .Helpers import get_Rm_and_Rc_from_MCS, calc_block_params


class RxDbb():

    def __init__(self):
        pass

    def set_STF_ingress_block(self, block):
        self.STF_ingress_block = block

    def set_CES_ingress_block(self, block):
        self.CES_ingress_block = block

    def set_PAY_ingress_block(self, block):
        self.PAY_ingress_block = block


class RxDbb_Flavour3Ideal(RxDbb):
    """Flavour 3, but without delays. """

    def __init__(self, env, log_path, log_subdir, MCS, number_of_blocks, number_of_block_padding_bits, aggregated_ppdus):

        modulation_rate, code_rate = get_Rm_and_Rc_from_MCS(MCS)

        detection_and_coarse_estiamtion_block = Block.DetectionAndCoarseEstimation(
            block_id=0, env=env, delay=Delay.Delay(0, 0), func=Func.Blank(), log_path=log_path, log_subdir=log_subdir,
            has_input_flag_buffer=False)

        joint_CFO_and_IQ_imbalance_estimation_block = Block.JointCFOAndIQImbalanceEstimation(
            block_id=1, env=env, delay=Delay.Delay(0, 0), func=Func.Blank(), log_path=log_path, log_subdir=log_subdir)

        noise_and_channel_estimation = Block.NoiseAndChannelEstimation(
            block_id=2, env=env, delay=Delay.Delay(0, 0), func=Func.Blank(), log_path=log_path, log_subdir=log_subdir)

        joint_IQ_imbalance_and_CFO_compensation_block = Block.JointIQImbalanceAndCFOCompensation(
            block_id=3, env=env, activation_seq_len=number_of_blocks * 512 * (aggregated_ppdus+1), delay=Delay.Delay(0, 0), func=Func.Blank(), log_path=log_path, log_subdir=log_subdir)

        agg_CFO_compensation_block = Block.AggregationBlankBlock(
            block_id=4, name='Agg CFO compensation', env=env, agg_len=512, log_path=log_path, log_subdir=log_subdir)

        frequency_domain_channel_equalization = Block.FrequencyDomainChannelEqualization(
            block_id=5, env=env, activation_seq_len=number_of_blocks * (aggregated_ppdus+1), delay=Delay.FDEAhmed(), func=Func.FDE(number_of_blocks, number_of_block_padding_bits), log_path=log_path, log_subdir=log_subdir)

        seg_FDE_block = Block.SegregationBlankBlock(
            block_id=6, name='Seg FDE', env=env, log_path=log_path, log_subdir=log_subdir)

        demapping_block = Block.Demapper(
            block_id=8, env=env, func=Func.DummyDemapper(modulation_rate),
            delay=Delay.Delay(0, 0), log_path=log_path,
            log_subdir=log_subdir)

        agg_demapping_block = Block.AggregationBlankBlock(
            block_id=9, name='Agg demapper', env=env, agg_len=672, log_path=log_path, log_subdir=log_subdir)

        decoding_block = Block.Decoder(
            block_id=10, env=env, func=Func.DummyDecoder(code_rate), delay=Delay.Delay(0, 0), log_path=log_path,
            log_subdir=log_subdir)

        seg_decoding_block = Block.SegregationBlankBlock(
            block_id=11, name='Seg decoder', env=env, log_path=log_path, log_subdir=log_subdir)

        # descrambling_block = Block.Descrambler(
        #     block_id=12, env=env, func=Func.Descrambler(scrambler_seed), delay=Delay.DescramblerChen180nm(), log_path=log_path,
        #     log_subdir=log_subdir)


        # Preamble-based block outputs
        detection_and_coarse_estiamtion_block.output_flag_buffer = joint_CFO_and_IQ_imbalance_estimation_block.input_flag_buffer
        joint_CFO_and_IQ_imbalance_estimation_block.output_data_buffer = noise_and_channel_estimation.input_data_buffer
        joint_CFO_and_IQ_imbalance_estimation_block.output_flag_buffer = joint_IQ_imbalance_and_CFO_compensation_block.input_flag_buffer
        noise_and_channel_estimation.output_flag_buffer = frequency_domain_channel_equalization.input_flag_buffer

        # Payload-based blocks (channel effect compensation)
        joint_IQ_imbalance_and_CFO_compensation_block.output_flag_buffer = joint_IQ_imbalance_and_CFO_compensation_block.input_flag_buffer  # Self activation (once started, keep on running)
        joint_IQ_imbalance_and_CFO_compensation_block.output_data_buffer = agg_CFO_compensation_block.input_data_buffer
        agg_CFO_compensation_block.output_data_buffer = frequency_domain_channel_equalization.input_data_buffer
        frequency_domain_channel_equalization.output_flag_buffer = frequency_domain_channel_equalization.input_flag_buffer  # Self activation (once started, keep on running)
        frequency_domain_channel_equalization.output_data_buffer = seg_FDE_block.input_data_buffer
        seg_FDE_block.output_data_buffer = demapping_block.input_data_buffer
        # seg_FDE_block.output_data_buffer = descrambling_block.input_data_buffer

        # Payload-based blocks (data reconstruction)
        demapping_block.output_data_buffer = agg_demapping_block.input_data_buffer
        agg_demapping_block.output_data_buffer = decoding_block.input_data_buffer
        decoding_block.output_data_buffer = seg_decoding_block.input_data_buffer
        seg_decoding_block.output_data_buffer = Buffer.Buffer(env, 'Output', log_path, log_subdir)

        self.set_STF_ingress_block(detection_and_coarse_estiamtion_block)
        self.set_CES_ingress_block(joint_CFO_and_IQ_imbalance_estimation_block)
        self.set_PAY_ingress_block(joint_IQ_imbalance_and_CFO_compensation_block)

        # Aggregate blocks that will be used
        self.blocks = [
            detection_and_coarse_estiamtion_block,
            joint_CFO_and_IQ_imbalance_estimation_block,
            noise_and_channel_estimation,
            joint_IQ_imbalance_and_CFO_compensation_block,
            agg_CFO_compensation_block,
            frequency_domain_channel_equalization,
            seg_FDE_block,
            demapping_block,
            agg_demapping_block,
            decoding_block,
            seg_decoding_block
        ]


    def update_packet_params(self, modulation_rate, code_rate, number_of_blocks):
        self.blocks[3].activation_seq_len = number_of_blocks * 512
        self.blocks[5].activation_seq_len = number_of_blocks
        self.blocks[7].modulation_rate = modulation_rate
        self.blocks[9].code_rate = code_rate


class RxDbb_Flavour3(RxDbb):

    def __init__(self, env, log_path, log_subdir, MCS, number_of_blocks, number_of_block_padding_bits, aggregated_ppdus, decoder_iterations, demapper_delay_instance):

        modulation_rate, code_rate = get_Rm_and_Rc_from_MCS(MCS)

        detection_and_coarse_estiamtion_block = Block.DetectionAndCoarseEstimation(
            block_id=0, env=env, delay=Delay.Delay(0, 0), func=Func.Blank(), log_path=log_path, log_subdir=log_subdir,
            has_input_flag_buffer=False)

        joint_CFO_and_IQ_imbalance_estimation_block = Block.JointCFOAndIQImbalanceEstimation(
            block_id=1, env=env, delay=Delay.Delay(0, 0), func=Func.Blank(), log_path=log_path, log_subdir=log_subdir)

        noise_and_channel_estimation = Block.NoiseAndChannelEstimation(
            block_id=2, env=env, delay=Delay.FFTAhmed(), func=Func.Blank(), log_path=log_path, log_subdir=log_subdir)

        joint_IQ_imbalance_and_CFO_compensation_block = Block.JointIQImbalanceAndCFOCompensation(
            block_id=3, env=env, activation_seq_len=number_of_blocks * 512 * (aggregated_ppdus+1), delay=Delay.Delay(0, 0), func=Func.Blank(), log_path=log_path, log_subdir=log_subdir)

        agg_CFO_compensation_block = Block.AggregationBlankBlock(
            block_id=4, name='Agg CFO compensation', env=env, agg_len=512, log_path=log_path, log_subdir=log_subdir)

        frequency_domain_channel_equalization = Block.FDE_HC(
            block_id=5, env=env, activation_seq_len=number_of_blocks * (aggregated_ppdus+1), delay=Delay.FDEAhmed(), func=Func.FDE(number_of_blocks, number_of_block_padding_bits))

        seg_FDE_block = Block.SegregationBlankBlock(
            block_id=6, name='Seg FDE', env=env, log_path=log_path, log_subdir=log_subdir)

        demapping_block = Block.Demapper(
            block_id=8, env=env, func=Func.DummyDemapper(modulation_rate),
            delay=demapper_delay_instance(modulation_rate), log_path=log_path,
            log_subdir=log_subdir)

        agg_demapping_block = Block.AggregationBlankBlock(
            block_id=9, name='Agg demapper', env=env, agg_len=672, log_path=log_path, log_subdir=log_subdir)

        decoding_block = Block.Decoder(
            block_id=10, env=env, func=Func.DummyDecoder(code_rate), delay=Delay.DecoderLiCustom(code_rate, decoder_iterations), log_path=log_path,
            log_subdir=log_subdir)

        seg_decoding_block = Block.SegregationBlankBlock(
            block_id=11, name='Seg decoder', env=env, log_path=log_path, log_subdir=log_subdir)

        # descrambling_block = Block.Descrambler(
        #     block_id=12, env=env, func=Func.Descrambler(scrambler_seed), delay=Delay.DescramblerChen180nm(), log_path=log_path,
        #     log_subdir=log_subdir)


        # Preamble-based block outputs
        detection_and_coarse_estiamtion_block.output_flag_buffer = joint_CFO_and_IQ_imbalance_estimation_block.input_flag_buffer
        joint_CFO_and_IQ_imbalance_estimation_block.output_data_buffer = noise_and_channel_estimation.input_data_buffer
        joint_CFO_and_IQ_imbalance_estimation_block.output_flag_buffer = joint_IQ_imbalance_and_CFO_compensation_block.input_flag_buffer
        noise_and_channel_estimation.output_flag_buffer = frequency_domain_channel_equalization.input_flag_buffer

        # Payload-based blocks (channel effect compensation)
        joint_IQ_imbalance_and_CFO_compensation_block.output_flag_buffer = joint_IQ_imbalance_and_CFO_compensation_block.input_flag_buffer  # Self activation (once started, keep on running)
        joint_IQ_imbalance_and_CFO_compensation_block.output_data_buffer = agg_CFO_compensation_block.input_data_buffer
        agg_CFO_compensation_block.output_data_buffer = frequency_domain_channel_equalization.input_data_buffer
        frequency_domain_channel_equalization.output_flag_buffer = frequency_domain_channel_equalization.input_flag_buffer  # Self activation (once started, keep on running)
        frequency_domain_channel_equalization.output_data_buffer = seg_FDE_block.input_data_buffer
        seg_FDE_block.output_data_buffer = demapping_block.input_data_buffer
        # seg_FDE_block.output_data_buffer = descrambling_block.input_data_buffer

        # Payload-based blocks (data reconstruction)
        demapping_block.output_data_buffer = agg_demapping_block.input_data_buffer
        agg_demapping_block.output_data_buffer = decoding_block.input_data_buffer
        decoding_block.output_data_buffer = seg_decoding_block.input_data_buffer
        seg_decoding_block.output_data_buffer = Buffer.Buffer(env, 'Output', log_path, log_subdir)

        self.set_STF_ingress_block(detection_and_coarse_estiamtion_block)
        self.set_CES_ingress_block(joint_CFO_and_IQ_imbalance_estimation_block)
        self.set_PAY_ingress_block(joint_IQ_imbalance_and_CFO_compensation_block)

        # Aggregate blocks that will be used
        self.blocks = [
            detection_and_coarse_estiamtion_block,
            joint_CFO_and_IQ_imbalance_estimation_block,
            noise_and_channel_estimation,
            joint_IQ_imbalance_and_CFO_compensation_block,
            agg_CFO_compensation_block,
            frequency_domain_channel_equalization,
            seg_FDE_block,
            demapping_block,
            agg_demapping_block,
            decoding_block,
            seg_decoding_block
        ]


    def update_packet_params(self, modulation_rate, code_rate, number_of_blocks):
        self.blocks[3].activation_seq_len = number_of_blocks * 512
        self.blocks[5].activation_seq_len = number_of_blocks
        self.blocks[7].modulation_rate = modulation_rate
        self.blocks[9].code_rate = code_rate


class RxDbb_Flavour2(RxDbb):

    def __init__(self, log_path, log_subdir, env, MCS, number_of_blocks, number_of_block_padding_bits, number_of_codewords, number_of_cw_padding_bits, decoder_iterations, demapper_delay_instance ):

        modulation_rate, code_rate = get_Rm_and_Rc_from_MCS(MCS)

        detection_and_coarse_estiamtion_block = Block.DetectionAndCoarseEstimation(
            block_id=0,
            env=env,
            delay=Delay.Delay(0, 0),
            func=Func.Blank(),
            log_path=log_path,
            log_subdir=log_subdir,
            has_input_flag_buffer=False
        )

        joint_CFO_and_IQ_imbalance_estimation_block = Block.JointCFOAndIQImbalanceEstimation(
            block_id=1,
            env=env,
            delay=Delay.Delay(0, 0),
            func=Func.Blank(),
            log_path=log_path,
            log_subdir=log_subdir
        )

        noise_and_channel_estimation = Block.NoiseAndChannelEstimation(
            block_id=2,
            env=env,
            delay=Delay.FFTAhmed(),
            func=Func.Blank(),
            log_path=log_path,
            log_subdir=log_subdir
        )

        joint_IQ_imbalance_and_CFO_compensation_block = Block.JointIQImbalanceAndCFOCompensation(
            block_id=3,
            env=env,
            activation_seq_len=number_of_blocks * 512,
            delay=Delay.Delay(0, 0),
            func=Func.Blank(),
            log_path=log_path,
            log_subdir=log_subdir
        )

        agg_CFO_compensation_block = Block.AggregationBlankBlock(
            block_id=4,
            name='Agg CFO compensation',
            env=env,
            agg_len=512,
            log_path=log_path,
            log_subdir=log_subdir
        )

        frequency_domain_channel_equalization = Block.FrequencyDomainChannelEqualization(
            block_id=5,
            env=env,
            activation_seq_len=number_of_blocks,
            delay=Delay.FDEAhmed(),
            func=Func.FDE(number_of_blocks, number_of_block_padding_bits),
            log_path=log_path, log_subdir=log_subdir
        )

        seg_FDE_block = Block.SegregationBlankBlock(
            block_id=6,
            name='Seg FDE',
            env=env,
            log_path=log_path,
            log_subdir=log_subdir
        )

        demapping_block = Block.Demapper(
            block_id=7,
            env=env,
            func=Func.Demapper(modulation_rate),
            delay=demapper_delay_instance(modulation_rate),
            log_path=log_path,
            log_subdir=log_subdir
        )

        agg_demapping_block = Block.AggregationBlankBlock(
            block_id=8,
            name='Agg demapper',
            env=env,
            agg_len=672,
            log_path=log_path,
            log_subdir=log_subdir
        )

        decoding_block = Block.Decoder(
            block_id=9,
            env=env,
            func=Func.Decoder(code_rate, number_of_codewords, number_of_cw_padding_bits),
            delay=Delay.DecoderLiCustom(code_rate, decoder_iterations),
            log_path=log_path,
            log_subdir=log_subdir
        )

        seg_decoding_block = Block.SegregationBlankBlock(
            block_id=10,
            name='Seg decoder',
            env=env,
            log_path=log_path,
            log_subdir=log_subdir
        )

        descrambling_block = Block.Descrambler(
            block_id=11,
            env=env,
            func=Func.Blank(),
            delay=Delay.DescramblerChen180nm(),
            log_path=log_path,
            log_subdir=log_subdir
        )

        output_block = Block.FlaglessBlock(
            block_id=12,
            name='output',
            func=Func.Blank(),
            delay=Delay.Delay(0,0),
            env=env,
            log_path=log_path,
            log_subdir=log_subdir
        )


        # Preamble-based block outputs
        detection_and_coarse_estiamtion_block.output_flag_buffer =          joint_CFO_and_IQ_imbalance_estimation_block.input_flag_buffer
        joint_CFO_and_IQ_imbalance_estimation_block.output_data_buffer =    noise_and_channel_estimation.input_data_buffer
        joint_CFO_and_IQ_imbalance_estimation_block.output_flag_buffer =    joint_IQ_imbalance_and_CFO_compensation_block.input_flag_buffer
        noise_and_channel_estimation.output_flag_buffer =                   frequency_domain_channel_equalization.input_flag_buffer

        # Payload-based blocks (channel effect compensation)
        joint_IQ_imbalance_and_CFO_compensation_block.output_flag_buffer =  joint_IQ_imbalance_and_CFO_compensation_block.input_flag_buffer  # Self activation (once started, keep on running)
        joint_IQ_imbalance_and_CFO_compensation_block.output_data_buffer =  agg_CFO_compensation_block.input_data_buffer
        agg_CFO_compensation_block.output_data_buffer =                     frequency_domain_channel_equalization.input_data_buffer
        frequency_domain_channel_equalization.output_flag_buffer =          frequency_domain_channel_equalization.input_flag_buffer  # Self activation (once started, keep on running)
        frequency_domain_channel_equalization.output_data_buffer =          seg_FDE_block.input_data_buffer
        seg_FDE_block.output_data_buffer =                                  demapping_block.input_data_buffer

        # Payload-based blocks (data reconstruction)
        demapping_block.output_data_buffer =                                agg_demapping_block.input_data_buffer
        agg_demapping_block.output_data_buffer =                            decoding_block.input_data_buffer
        decoding_block.output_data_buffer =                                 seg_decoding_block.input_data_buffer
        seg_decoding_block.output_data_buffer =                             descrambling_block.input_data_buffer
        descrambling_block.output_data_buffer =                             output_block.input_data_buffer

        self.set_STF_ingress_block(detection_and_coarse_estiamtion_block)
        self.set_CES_ingress_block(joint_CFO_and_IQ_imbalance_estimation_block)
        self.set_PAY_ingress_block(joint_IQ_imbalance_and_CFO_compensation_block)

        # Aggregate blocks that will be used
        self.blocks = [
            detection_and_coarse_estiamtion_block,
            joint_CFO_and_IQ_imbalance_estimation_block,
            noise_and_channel_estimation,
            joint_IQ_imbalance_and_CFO_compensation_block,
            agg_CFO_compensation_block,
            frequency_domain_channel_equalization,
            seg_FDE_block,
            demapping_block,
            agg_demapping_block,
            decoding_block,
            seg_decoding_block,
            descrambling_block,
            output_block
        ]


    def update_packet_params(self, modulation_rate, code_rate, number_of_blocks):
        self.blocks[3].activation_seq_len = number_of_blocks * 512
        self.blocks[5].activation_seq_len = number_of_blocks
        self.blocks[7].modulation_rate = modulation_rate
        self.blocks[9].code_rate = code_rate



class RxDbb_Flavour1(RxDbb):

    def __init__(self, env, log_path, log_subdir, code_rate, modulation_rate, number_of_blocks):

        detection_and_coarse_estiamtion_block = Block.DetectionAndCoarseEstimation(
            block_id=0, env=env, delay=Delay.Delay(0, 2500), log_path=log_path, log_subdir=log_subdir,
            has_input_flag_buffer=False)
        IQ_imbalance_estimation_block = Block.IQImbalanceEstimation(
            block_id=1, env=env, delay=Delay.Delay(0, 0), log_path=log_path, log_subdir=log_subdir)
        fine_CFO_estimation_block = Block.Finefine_CFO_estimation_blockstimation(
            block_id=2, env=env, delay=Delay.Delay(0, 0), log_path=log_path, log_subdir=log_subdir,
            has_input_flag_buffer=False)
        channel_estimation_block = Block.ChannelEstimation(
            block_id=3, env=env, delay=Delay.Delay(0, 0), log_path=log_path, log_subdir=log_subdir,
            has_input_flag_buffer=False)
        noise_estimation_block = Block.NoiseEstimation(
            block_id=4, env=env, delay=Delay.Delay(0, 0), log_path=log_path, log_subdir=log_subdir,
            has_input_flag_buffer=False)

        IQ_imbalance_compensation_block = Block.IQImbalanceCompensation(
            block_id=5, env=env, activation_seq_len=number_of_blocks * 512, delay=Delay.Delay(0, 0),
            log_path=log_path, log_subdir=log_subdir)
        CFO_compensation_block = Block.CFOCompensation(
            block_id=6, env=env, activation_seq_len=number_of_blocks * 512, delay=Delay.Delay(0, 0),
            log_path=log_path, log_subdir=log_subdir)
        agg_CFO_compensation_block = Block.AggregationBlankBlock(
            block_id=7, name='Agg CFO compensation', env=env, agg_len=512, log_path=log_path, log_subdir=log_subdir)
        fft = Block.FFT(
            block_id=8, env=env, func=Func.Blank(), delay=Delay.FFTAhmed(100), log_path=log_path, log_subdir=log_subdir)
        channel_equalisation_block = Block.ChannelEqualisation(
            block_id=9, activation_seq_len=number_of_blocks, env=env,
            delay=Delay.EqualizerCustom(fft.delay.clock_MHz), log_path=log_path, log_subdir=log_subdir)
        ifft = Block.IFFT(
            block_id=10, env=env, func=Func.Blank(), delay=Delay.FFTAhmed(), log_path=log_path, log_subdir=log_subdir)
        pilot_based_tracking_block = Block.PilotBasedTracking(
            block_id=11, env=env, func=Func.PilotBasedTracking(), delay=Delay.Delay(0, 0), log_path=log_path,
            log_subdir=log_subdir)
        seg_pilot_based_tracking_block = Block.SegregationBlankBlock(
            block_id=12, name='Seg pilot-based tracking', env=env, log_path=log_path, log_subdir=log_subdir)
        phase_correction_block = Block.PhaseCorrection(
            block_id=13, env=env, func=Func.PhaseCorrection(), delay=Delay.Delay(0, 0), log_path=log_path,
            log_subdir=log_subdir)

        demapping_block = Block.Demapper(
            block_id=14, env=env, func=Func.Demapper(),
            delay=Delay.DemapperJafriSuboptimalConventionalFPGAtoASIC65nm(modulation_rate), log_path=log_path,
            log_subdir=log_subdir)
        agg_demapping_block = Block.AggregationBlankBlock(
            block_id=15, name='Agg demapper', env=env, agg_len=672, log_path=log_path, log_subdir=log_subdir)
        # decoding_block = Block.Decoder(
        #     env=env, func=Func.Decoder(code_rate), delay=Delay.DecoderWiener65nm(code_rate) )
        decoding_block = Block.Decoder(
            block_id=16, env=env, func=Func.Decoder(code_rate), delay=Delay.DecoderLi_1_2(code_rate), log_path=log_path,
            log_subdir=log_subdir)
        seg_decoding_block = Block.SegregationBlankBlock(
            block_id=17, name='Seg decoder', env=env, log_path=log_path, log_subdir=log_subdir)
        descrambling_block = Block.Descrambler(
            block_id=18, env=env, func=Func.Blank(), delay=Delay.DescramblerChen180nm(), log_path=log_path,
            log_subdir=log_subdir)

        # Preamble-based block outputs
        detection_and_coarse_estiamtion_block.output_flag_buffer = IQ_imbalance_estimation_block.input_flag_buffer
        IQ_imbalance_estimation_block.output_data_buffer = fine_CFO_estimation_block.input_data_buffer
        IQ_imbalance_estimation_block.output_flag_buffer = IQ_imbalance_compensation_block.input_flag_buffer
        fine_CFO_estimation_block.output_flag_buffer = CFO_compensation_block.input_flag_buffer
        fine_CFO_estimation_block.output_data_buffer = channel_estimation_block.input_data_buffer
        channel_estimation_block.output_flag_buffer = channel_equalisation_block.input_flag_buffer
        channel_estimation_block.output_data_buffer = noise_estimation_block.input_data_buffer

        # Payload-based blocks (channel effect compensation)
        IQ_imbalance_compensation_block.output_flag_buffer = IQ_imbalance_compensation_block.input_flag_buffer  # Self activation (once started, keep on running)
        IQ_imbalance_compensation_block.output_data_buffer = CFO_compensation_block.input_data_buffer
        CFO_compensation_block.output_flag_buffer = CFO_compensation_block.input_flag_buffer  # Self activation (once started, keep on running)
        CFO_compensation_block.output_data_buffer = agg_CFO_compensation_block.input_data_buffer
        agg_CFO_compensation_block.output_data_buffer = fft.input_data_buffer
        fft.output_data_buffer = channel_equalisation_block.input_data_buffer
        channel_equalisation_block.output_flag_buffer = channel_equalisation_block.input_flag_buffer  # Self activation (once started, keep on running)
        channel_equalisation_block.output_data_buffer = ifft.input_data_buffer
        ifft.output_data_buffer = pilot_based_tracking_block.input_data_buffer
        pilot_based_tracking_block.output_data_buffer = seg_pilot_based_tracking_block.input_data_buffer
        seg_pilot_based_tracking_block.output_data_buffer = phase_correction_block.input_data_buffer
        phase_correction_block.output_data_buffer = demapping_block.input_data_buffer

        # Payload-based blocks (data reconstruction)
        demapping_block.output_data_buffer = agg_demapping_block.input_data_buffer
        agg_demapping_block.output_data_buffer = decoding_block.input_data_buffer
        decoding_block.output_data_buffer = seg_decoding_block.input_data_buffer
        seg_decoding_block.output_data_buffer = descrambling_block.input_data_buffer

        self.set_STF_ingress_block(detection_and_coarse_estiamtion_block)
        self.set_CES_ingress_block(IQ_imbalance_estimation_block)
        self.set_PAY_ingress_block(IQ_imbalance_compensation_block)

        # Aggregate blocks that will be used
        self.blocks = [
            detection_and_coarse_estiamtion_block,
            IQ_imbalance_estimation_block,
            fine_CFO_estimation_block,
            channel_estimation_block,
            noise_estimation_block,
            IQ_imbalance_compensation_block,
            CFO_compensation_block,
            agg_CFO_compensation_block,
            fft,
            channel_equalisation_block,
            ifft,
            pilot_based_tracking_block,
            seg_pilot_based_tracking_block,
            phase_correction_block,
            demapping_block,
            agg_demapping_block,
            decoding_block,
            seg_decoding_block,
            descrambling_block
        ]


### Simplified RX DBB ##################################################################################################

class RxDbbSimplified():

    def __init__(self, log_path, log_subdir, env, MCS, num_of_blocks, block_padding, decoder_iterations, demapper_delay_instance):

        modulation_rate, code_rate = get_Rm_and_Rc_from_MCS(MCS)

        noise_and_channel_estimation = BlockSimplified.CEST_Simple(
            block_id=0,
            env=env,
            log_path=log_path,
            log_subdir=log_subdir,
            delay=Delay.FFTAhmed()
        )

        frequency_domain_channel_equalization = BlockSimplified.FDE_Simple(
            block_id=1,
            env=env,
            log_path=log_path,
            log_subdir=log_subdir,
            delay=Delay.EqualizerCustom(),
            func=Func.FDE(num_of_blocks, block_padding)
        )

        demapping_block = BlockSimplified.Demapper_Simple(
            block_id=2,
            env=env,
            log_path=log_path,
            log_subdir=log_subdir,
            delay=demapper_delay_instance(modulation_rate),
            func=Func.DummyDemapper(modulation_rate)
        )

        decoding_block = BlockSimplified.Decoder_Simple(
            block_id=3,
            env=env,
            log_path=log_path,
            log_subdir=log_subdir,
            delay=Delay.DecoderLiCustom(code_rate, decoder_iterations),
            func=Func.DummyDecoder(code_rate)
        )

        # Preamble-based block outputs
        self.STF_ingress_block = None
        self.CES_ingress_block = noise_and_channel_estimation
        noise_and_channel_estimation.output_flag_buffer = frequency_domain_channel_equalization.input_flag_buffer

        # Payload-based blocks
        self.PAY_ingress_block = frequency_domain_channel_equalization
        frequency_domain_channel_equalization.output_flag_buffer = frequency_domain_channel_equalization.input_flag_buffer  # Self activation (once started, keep on running)
        frequency_domain_channel_equalization.output_data_buffer = demapping_block.input_data_buffer
        demapping_block.output_data_buffer = decoding_block.input_data_buffer

        # Aggregate blocks that will be used
        self.blocks = [
            noise_and_channel_estimation,
            frequency_domain_channel_equalization,
            demapping_block,
            decoding_block
        ]


class RxDbbSimplified448():

    def __init__(self, log_path, log_subdir, env, MCS, num_of_blocks, block_padding, number_of_codewords, number_of_cw_padding_bits, decoder_iterations, demapper_delay_instance):

        modulation_rate, code_rate = get_Rm_and_Rc_from_MCS(MCS)

        noise_and_channel_estimation = Block.CEST_Simple(
            block_id=0,
            env=env,
            log_path=log_path,
            log_subdir=log_subdir,
            delay=Delay.FFTAhmed()
        )

        frequency_domain_channel_equalization = Block.FDE_Simple448(
            block_id=1,
            env=env,
            log_path=log_path,
            log_subdir=log_subdir,
            delay=Delay.EqualizerCustom(),
            func=Func.FDE(num_of_blocks, block_padding)
        )

        demapping_block = Block.Demapper_Simple448(
            block_id=2,
            env=env,
            log_path=log_path,
            log_subdir=log_subdir,
            delay=demapper_delay_instance(modulation_rate),
            func=Func.Demapper(modulation_rate)
        )

        decoding_block = Block.Decoder_Simple(
            block_id=3,
            env=env,
            log_path=log_path,
            log_subdir=log_subdir,
            delay=Delay.DecoderLiCustom(code_rate, decoder_iterations),
            func=Func.Decoder(code_rate, number_of_codewords, number_of_cw_padding_bits)
        )

        # Preamble-based block outputs
        self.STF_ingress_block = None
        self.CES_ingress_block = noise_and_channel_estimation
        noise_and_channel_estimation.output_flag_buffer = frequency_domain_channel_equalization.input_flag_buffer

        # Payload-based blocks
        self.PAY_ingress_block = frequency_domain_channel_equalization
        frequency_domain_channel_equalization.output_flag_buffer = frequency_domain_channel_equalization.input_flag_buffer  # Self activation (once started, keep on running)
        frequency_domain_channel_equalization.output_data_buffer = demapping_block.input_data_buffer
        demapping_block.output_data_buffer = decoding_block.input_data_buffer

        # Aggregate blocks that will be used
        self.blocks = [
            noise_and_channel_estimation,
            frequency_domain_channel_equalization,
            demapping_block,
            decoding_block
        ]
