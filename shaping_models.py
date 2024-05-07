import torch
import mokka
from mokka import utils
from torch.serialization import storage_to_tensor_type
import numpy as np
import settings




class AWGNAutoencoder:
    def __init__(self):
        self.SNR = 11
        self.m = 4
        self.nsymbols = 2**16

        N0 = torch.tensor(utils.N0(self.SNR))

        self.mapper = mokka.mapping.torch.ConstellationMapper(self.m, qam_init=True).to(
            device
        )
        self.demapper = mokka.mapping.torch.ConstellationDemapper(self.m).to(device)
        self.channel = mokka.channels.torch.ComplexAWGN(N0).to(device)
        self.optim = torch.optim.Adam(
            (*self.mapper.parameters(), *self.demapper.parameters()), lr=1e-3
        )

    def step(self):
        bits = utils.generators.torch.generate_bits((self.nsymbols, self.m))
        symbols = self.mapper(bits).flatten()
        tx_signal = symbols
        rx_signal = self.channel.forward(tx_signal)
        llrs = self.demapper(rx_signal.flatten()[:, None])
        bmi = mokka.inft.torch.BMI(
            self.m,
            self.nsymbols,
            bits,
            llrs,
        )
        loss = self.m - bmi
        loss.backward()
        self.optim.step()

        self.optim.zero_grad()
        return bmi


class ShapingAutoencoder:
    config: dict | None = None

    def __init__(self, config):
        """
        Configure the Shaping from the settings dictionary.
        """

        # Define sets of settings which should trigger a change of the corresponding component
        self.mapper_settings = ("bits_per_symbol", "type", "qam_init")
        self.channel_settings = ("channel", "SNR", "LW", "symbols_per_step")
        self.cpe_settings = ("cpe",)
        self.demapper_settings = ("demapper", "bits_per_symbol", "objective", "type")

        self.update_config(config)

    def check_config_change(self, attributes, config):
        if self.config is None:
            return True
        for att in attributes:
            if att not in self.config or self.config[att] != config[att]:
                return True
        return False

    def update_config(self, config):
        """
        Update saved settings with possibly new settings from a dictionary.

        Handle changes gracefully
        """
        if self.config is None:
            self.config = {}

        # We do a lot of checking here if a value that is used for this particular element
        # has changed. A bit cumbersome, probably there is a a smarter way to do that

        # Configure mapper
        if self.check_config_change(self.mapper_settings, config):
            self.mapper = mokka.mapping.torch.ConstellationMapper(
                config["bits_per_symbol"], qam_init=config["qam_init"]
            )
            self.pcs_sampler = mokka.mapping.torch.PCSSampler(
                config["bits_per_symbol"],
                l_init=torch.ones(2 ** config["bits_per_symbol"], dtype=torch.float),
            )
       # Configure cpe

        if self.check_config_change(self.cpe_settings, config):
            cpe_window_length = 50
            if config["cpe"] == settings.CPE.BPS:
                M_test_angles = 60
                diff = True
                n_sectors = 1
                diff_BPS_temp = 1e-3
                self.cpe = mokka.synchronizers.phase.torch.BPS(
                    M_test_angles,
                    self.mapper.get_constellation(),
                    cpe_window_length,
                    diff,
                    diff_BPS_temp,
                    n_sectors,
                    avg_filter_type="rect",
                )

            elif config["cpe"] == settings.CPE.VV:
                self.cpe = mokka.synchronizers.phase.torch.vandv.ViterbiViterbi(
                    window_length=cpe_window_length
                )


        # Configure channel
        if self.check_config_change(self.channel_settings, config):
            self.channel_chain = []
            N0 = torch.as_tensor(utils.N0(config["SNR"]))
            if config["channel"] == settings.ShapingChannel.AWGN:
                self.channel_chain.append(mokka.channels.torch.ComplexAWGN(N0))
            elif config["channel"] == settings.ShapingChannel.Wiener:
                sigma_phi = utils.sigma_phi(config["LW"], config["symbol_rate"])
                pn_channel = mokka.channels.torch.PhasenoiseWiener(
                    start_phase_init=0, start_phase_width=0
                )
                self.channel_chain.append(lambda syms: pn_channel(syms, N0, sigma_phi))
            elif config["channel"] == settings.ShapingChannel.Optical:
                rrc_len = 101
                rrc_rolloff = 0.1
                n_channels = 11
                wdm_spacing = 100e9
                R_sym = 32e9
                n_up1 = 5  # 160 GHz
                n_up2 = 10  # 1600 GHz
                rrc = mokka.pulseshaping.torch.PulseShaping.get_rrc_ir(
                    rrc_len, rrc_rolloff, n_up1
                )
                mux = mokka.channels.torch.WDMMux(
                    n_channels, wdm_spacing, R_sym * n_up1, R_sym * n_up1 * n_up2
                )

                # SSFM Channel parameters
                n_spans = 3
                length_span = 100  # km
                nz = 100  # steps per span
                edfa_mode = "alpha_equalization"
                edfa_noise = True
                edfa_noisefigure = 5  # [dB]
                wavelength = 1550  # [nm]
                carrier_frequency = 3e8 / wavelength  # [GHz]
                launch_power = torch.nn.Parameter(torch.tensor(7.0))  # [dBm]
                P_input_lin = (10 ** (launch_power / 10)) * 1e-3
                bw = R_sym * n_up1 * n_up2

                alphadb = torch.tensor(0.2)  # Dämpfung [dB/km]
                D = torch.tensor(17.0)  # [ps/nm/km] = [μs/m²]
                gamma = torch.tensor(1.7)  # [1/W/km]

                # alpha = torch.tensor(alphadb * (np.log(10) / 10), dtype=torch.float32)
                beta2 = torch.tensor(mokka.utils.beta2(D, wavelength))  # ps**2/km

                z_length = torch.tensor(n_spans * length_span)  # [km]
                dz = torch.tensor(length_span / nz)  # [km]
                padding = 0

                # SSFM
                edfa_amp_torch = mokka.channels.torch.EDFAAmpSinglePol(
                    length_span,
                    edfa_mode,
                    alphadb,
                    edfa_noise,
                    edfa_noisefigure,
                    carrier_frequency,
                    bw,
                    P_input_lin,
                    padding,
                )

                ssfm_channel = mokka.channels.torch.SSFMPropagationSinglePol(
                    1 / bw,
                    dz,
                    alphadb,
                    torch.tensor([0, 0, beta2]),
                    gamma,
                    length_span,
                    n_spans,
                    amp=edfa_amp_torch,
                    solver_method="fixed",
                )

                # Receiver
                wdm_demux_channels = 16
                n_down = n_up1

                wdm_used_channels = torch.arange(-5, 5 + 1, dtype=torch.int)
                demux = mokka.channels.torch.WDMDemux(
                    wdm_demux_channels,
                    wdm_spacing,
                    R_sym,
                    R_sym * n_up1 * n_up2,
                    wdm_used_channels,
                    method="polyphase",
                )  # This will give channels of 100 GHz width with a polyphase filterbank

                def resample_and_MF(signals):
                    results = []
                    for signal in signals:
                        signal = mokka.channels.torch.downsample(
                            n_up2,
                            mokka.channels.torch.upsample(
                                wdm_demux_channels,
                                signal,
                                filter_gain=np.sqrt(wdm_demux_channels),
                            ),
                            filter_gain=np.sqrt(n_up2),
                        )
                        results.append(rrc.matched(signal, n_down).unsqueeze(0))
                    return torch.cat(results, dim=0)

                # Apply CD compensation after channelization
                cd_compensation = mokka.equalizers.torch.CD_compensation(
                    1 / bw, beta2, z_length
                )
                # Apply Phase compensation after matched filter
                bps_testangles = 40
                bps_symbols = self.mapper.get_constellation()
                bps_avg_length = 120
                bps_diff = True
                bps_no_sectors = 4
                bps_avg_filter_type = "rect"
                phase_comp = self.cpe
                def propagate_optical_channel(tx_symbols):
                    # Generate TX WDM Signal
                    tx_signal = torch.zeros(
                        (
                            n_channels,
                            (config["symbols_per_step"] + 2 * rrc_len) * n_up1,
                        ),
                        dtype=torch.complex64,
                    )
                    for idx, sym in enumerate(tx_symbols):
                        tx_signal[idx] = rrc(sym, n_up1)
                    wdm_signal = mux(tx_signal)

                    P_input_lin = (10 ** (launch_power / 10)) * 1e-3
                    gain = torch.sqrt(
                        P_input_lin / (torch.mean(torch.abs(wdm_signal) ** 2))
                    )
                    wdm_signal = wdm_signal * gain
                    wdm_rx_signal = ssfm_channel(wdm_signal)
                    wdm_rx_signal_cd = cd_compensation(wdm_rx_signal)
                    rx_signal = demux(wdm_rx_signal_cd) / gain

                    rx_symbols = resample_and_MF(rx_signal)
                    phase_syms = []
                    for syms in rx_symbols:
                        phase_syms.append(phase_comp(syms)[0].unsqueeze(0))
                    rx_symbols = torch.cat(phase_syms, dim=0)
                    return rx_symbols

                self.channel_chain = []
                self.channel_chain.append(propagate_optical_channel)


        # Configure demapper
        if self.check_config_change(self.demapper_settings, config):
            bitwise = config["objective"] == settings.ShapingObjective.BMI
            if config["demapper"] == settings.Demapper.Neural:
                self.demapper = mokka.mapping.torch.ConstellationDemapper(
                    m=config["bits_per_symbol"], bitwise=bitwise, with_logit=bitwise
                )
                # self.optim.add_param_group(
                #     {"params": self.demapper.parameters(), "lr": config["lr"]}
                # )
            elif config["demapper"] == settings.Demapper.Gaussian:
                N0 = torch.as_tensor(utils.N0(config["SNR"]))
                self.demapper = mokka.mapping.torch.ClassicalDemapper(
                    noise_sigma=torch.sqrt(N0),
                    constellation=self.mapper.get_constellation(),
                    bitwise=bitwise,
                )

        # Configure optimizer and specify optimization parameters
        if config["type"] == settings.ShapingType.Geometric:
            optim_params = (*self.mapper.parameters(), *self.demapper.parameters())
        elif config["type"] == settings.ShapingType.Joint:
            optim_params = (
                *self.mapper.parameters(),
                *self.demapper.parameters(),
                *self.pcs_sampler.parameters(),
            )
        elif config["type"] == settings.ShapingType.Probabilistic:
            optim_params = (*self.pcs_sampler.parameters(), *self.demapper.parameters())

        self.optim = torch.optim.Adam(optim_params, lr=1e-3)
        self.config = config

    def channel(self, tx_signal):
        for ch in self.channel_chain:
            tx_signal = ch(tx_signal)
        return tx_signal

    def step(self):
        results = {}
        if self.config is None:
            return

        # Reconfigure mapper & demapper with probabilities from pcs_sampler
        self.mapper.p_symbols = self.pcs_sampler.p_symbols()
        # Prepare for next run and update constellation in CPE and Demapper (if necessary)
        if self.config["cpe"] == settings.CPE.BPS:
            self.cpe.set_constellation(self.mapper.get_constellation())
        if self.config["demapper"] == settings.Demapper.Gaussian:
            self.demapper.p_symbols = self.pcs_sampler.p_symbols()
            self.demapper.update_constellation(self.mapper.get_constellation())

        if self.config["channel"] == settings.ShapingChannel.Optical:
            # we simulate 11 WDM channels
            symbol_idxs = self.pcs_sampler(
                self.config["symbols_per_step"] * 11
            ).reshape(11, -1)
            bits = torch.stack(
                tuple(
                    utils.bitops.torch.idx2bits(
                        channel_idxs, self.config["bits_per_symbol"]
                    )
                    for channel_idxs in symbol_idxs
                )
            )
            if self.config["objective"] == settings.ShapingObjective.MI:
                # We compute the one_hot vectors already outside the mapper
                one_hot = torch.stack(
                    tuple(mokka.utils.bitops.torch.bits_to_onehot(b) for b in bits)
                )
                symbols = torch.stack(
                    tuple(self.mapper(oh, one_hot=True).flatten() for oh in one_hot)
                )
            else:
                symbols = torch.stack(tuple(self.mapper(b).flatten() for b in bits))

        else:
            symbol_idxs = self.pcs_sampler(self.config["symbols_per_step"])
            bits = utils.bitops.torch.idx2bits(
                symbol_idxs, self.config["bits_per_symbol"]
            )
            if self.config["objective"] == settings.ShapingObjective.MI:
                # We compute the one_hot vectors already outside the mapper
                one_hot = mokka.utils.bitops.torch.bits_to_onehot(bits)
                symbols = self.mapper(one_hot, one_hot=True).flatten()
            else:
                symbols = self.mapper(bits).flatten()

        tx_signal = symbols
        rx_signal = self.channel(tx_signal.clone())
        results["tx_signal"] = tx_signal.detach().clone().cpu().flatten()
        results["rx_signal"] = rx_signal.detach().clone().cpu().flatten()
        if self.config["cpe"] == settings.CPE.BPS:
            rx_signal = self.cpe(rx_signal)[0]
        elif self.config["cpe"] == settings.CPE.VV:
            rx_signal = self.cpe(rx_signal)
        results["rx_signal_postcpe"] = rx_signal.detach().clone().cpu().flatten()

        if self.config["channel"] == settings.ShapingChannel.Optical:
            # Change between MI & GMI objective
            if self.config["objective"] == settings.ShapingObjective.BMI:
                llrs = torch.stack(
                    tuple(self.demapper(rxs.flatten()[:, None]) for rxs in rx_signal)
                )
                bmi = torch.stack(
                    tuple(
                        mokka.inft.torch.BMI(
                            self.config["bits_per_symbol"],
                            self.config["symbols_per_step"],
                            bits,
                            llrs_per_channel,
                            p=self.pcs_sampler.p_symbols(),
                        )
                        for llrs_per_channel in llrs
                    )
                )
                results["bmi"] = torch.mean(bmi.detach().clone().cpu())
                loss = self.config["bits_per_symbol"] - torch.mean(bmi)
            elif self.config["objective"] == settings.ShapingObjective.MI:
                q_values = torch.stack(
                    tuple(self.demapper(rxs.flatten()[:, None]) for rxs in rx_signal)
                )
                mi = torch.stack(
                    tuple(
                        mokka.inft.torch.MI(
                            2 ** self.config["bits_per_symbol"],
                            self.pcs_sampler.p_symbols().unsqueeze(0),
                            self.config["symbols_per_step"],
                            symbol_idxs,
                            q_values_per_channel,
                        )
                        for q_values_per_channel in q_values
                    )
                )
                results["mi"] = torch.mean(mi.detach().clone().cpu())
                loss = self.config["bits_per_symbol"] - torch.mean(mi)

        else:
            # Change between MI & GMI objective
            if self.config["objective"] == settings.ShapingObjective.BMI:
                llrs = self.demapper(rx_signal.flatten()[:, None])
                bmi = mokka.inft.torch.BMI(
                    self.config["bits_per_symbol"],
                    self.config["symbols_per_step"],
                    bits,
                    llrs,
                    p=self.pcs_sampler.p_symbols(),
                )
                results["bmi"] = bmi.detach().clone().cpu()
                loss = self.config["bits_per_symbol"] - bmi
            elif self.config["objective"] == settings.ShapingObjective.MI:
                q_values = self.demapper(rx_signal.flatten()[:, None])
                mi = mokka.inft.torch.MI(
                    2 ** self.config["bits_per_symbol"],
                    self.pcs_sampler.p_symbols().unsqueeze(0),
                    self.config["symbols_per_step"],
                    symbol_idxs,
                    q_values,
                )
                results["mi"] = mi.detach().clone().cpu()
                loss = self.config["bits_per_symbol"] - mi

        loss.backward()
        results["loss"] = loss.detach().clone().cpu()
        self.optim.step()
        self.optim.zero_grad()

        return results
