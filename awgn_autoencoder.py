import torch
import mokka
from mokka import utils


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
    def __init__(self, settings):
        """
        Configure the Shaping from the settings dictionary.
        """
        self.update_settings(settings)
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

    def update_settings(self, settings):
        """
        Update saved settings with possibly new settings from a dictionary.

        Handle changes gracefully
        """


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
