import torch
import mokka
from mokka import utils
from torch.serialization import storage_to_tensor_type
import settings


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)


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
        self.update_config(config)
        # self.SNR = 11
        # self.m = 4
        # self.nsymbols = 2**16

        # N0 = torch.tensor(utils.N0(self.SNR))

        # self.mapper = mokka.mapping.torch.ConstellationMapper(self.m, qam_init=True).to(
        #     device
        # )
        # self.demapper = mokka.mapping.torch.ConstellationDemapper(self.m).to(device)
        # self.channel = mokka.channels.torch.ComplexAWGN(N0).to(device)
        self.optim = torch.optim.Adam(
            (*self.mapper.parameters(), *self.demapper.parameters()), lr=1e-3
        )

    def update_config(self, config):
        """
        Update saved settings with possibly new settings from a dictionary.

        Handle changes gracefully
        """
        if self.config is None:
            self.config = {}

        # Configure mapper
        if (
            "bits_per_symbol" not in self.config
            or self.config["bits_per_symbol"] != config["bits_per_symbol"]
        ):
            self.mapper = mokka.mapping.torch.ConstellationMapper(
                config["bits_per_symbol"], qam_init=True
            )

        # Configure channel
        if (
            "channel" not in self.config
            or self.config["channel"] != config["channel"]
            or "SNR" not in self.config
            or self.config["SNR"] != config["SNR"]
            or "LW" not in self.config
            or self.config["LW"] != config["LW"]
        ):
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

        # Configure cpe

        if "cpe" not in self.config or self.config["cpe"] != config["cpe"]:
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

        # Configure demapper
        if (
            "demapper" not in self.config
            or self.config["demapper"] != config["demapper"]
            or self.config["bits_per_symbol"] != config["bits_per_symbol"]
        ):
            if config["demapper"] == settings.Demapper.Neural:
                self.demapper = mokka.mapping.torch.ConstellationDemapper(
                    m=config["bits_per_symbol"]
                )
                # self.optim.add_param_group(
                #     {"params": self.demapper.parameters(), "lr": config["lr"]}
                # )
            elif config["demapper"] == settings.Demapper.Gaussian:
                N0 = torch.as_tensor(utils.N0(config["SNR"]))
                self.demapper = mokka.mapping.torch.ClassicalDemapper(
                    noise_sigma=torch.sqrt(N0),
                    constellation=self.mapper.get_constellation(),
                )

        # Configure Objective
        if (
            "objective" not in self.config
            or self.config["objective"] != config["objective"]
        ):
            pass

        self.config = config
        self.optim = torch.optim.Adam(
            (*self.mapper.parameters(), *self.demapper.parameters()), lr=1e-3
        )

    def channel(self, tx_signal):
        for ch in self.channel_chain:
            tx_signal = ch(tx_signal)
        return tx_signal

    def step(self):
        results = {}
        if self.config is None:
            return
        bits = utils.generators.torch.generate_bits(
            (self.config["symbols_per_step"], self.config["bits_per_symbol"])
        )
        symbols = self.mapper(bits).flatten()
        tx_signal = symbols
        rx_signal = self.channel(tx_signal.clone())
        results["tx_signal"] = tx_signal.detach().clone().cpu()
        results["rx_signal"] = rx_signal.detach().clone().cpu()
        if self.config["cpe"] == settings.CPE.BPS:
            rx_signal = self.cpe(rx_signal)[0]
        elif self.config["cpe"] == settings.CPE.VV:
            rx_signal = self.cpe(rx_signal)
        results["rx_signal_postcpe"] = rx_signal.detach().clone().cpu()

        # Change between MI & GMI objective
        llrs = self.demapper(rx_signal.flatten()[:, None])
        bmi = mokka.inft.torch.BMI(
            self.config["bits_per_symbol"],
            self.config["symbols_per_step"],
            bits,
            llrs,
        )
        results["bmi"] = bmi.detach().clone().cpu()
        loss = self.config["bits_per_symbol"] - bmi
        loss.backward()
        results["loss"] = loss.detach().clone().cpu()
        self.optim.step()
        self.optim.zero_grad()
        # Prepare for next run and update constellation in CPE and Demapper (if necessary)
        if self.config["cpe"] == settings.CPE.BPS:
            self.cpe.set_constellation(self.mapper.get_constellation())
        if self.config["demapper"] == settings.Demapper.Gaussian:
            self.demapper.update_constellation(self.mapper.get_constellation())
        return results
