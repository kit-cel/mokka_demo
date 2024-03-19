################################################################################################
# Author:        Vincent Lauinger,
# Affiliation:   Communications Engineering Lab (CEL), Karlsruhe Institute of Technology (KIT)
# Contact:       vincent.lauinger@kit.edu
# Last revision: 15th of June 2022
################################################################################################

import numpy as np
from numpy.core.numeric import Inf
import torch as t
import torch
import torch.optim as optim

import mokka

import matplotlib.pyplot as plt

from mokka.equalizers.adaptive.torch import VAE_LE_DP

class EqualizerSimulation:
    config: dict | None = None
    current_frame: int = 0

    def __init__(self, config):
        self.update_config(config)

    def update_config(self, config):
        self.mod = "64-QAM"
        self.sps = 2
        self.SNR = 24
        self.nu = 0.0270955
        self.M = 25
        self.theta_diff = 0
        self.theta = 0
        self.lr_optim = 0.003
        self.batch_len = 200
        self.N_frame_max = 10000
        self.num_frames = 100
        self.flex_step = 10
        self.channel = "h0"
        self.symb_rate = 100000000000.0
        self.tau_cd = -2.6e-23
        self.tau_pmd = 5e-12
        self.phiIQ = torch.tensor([0 + 0j, 0 + 0j])
        self.N_lrhalf = 20

    def step(self):
        # This should process frame by frame and output the result so it can
        # be displayed in the GUI
        if self.current_frame == self.num_frames:
            return
        if self.current_frame == 0:
            (
                self.h_est,
                self.h_channel,
                self.P,
                self.amp_levels,
                self.amps,
                self.bit_arr,
                self.pol,
                self.nu_sc,
                self.var,
                self.pow_mean,
                self.kurtosis,
                self.H_P,
            ) = init(self.channel, self.mod, self.nu, self.sps, self.M, self.SNR)
            if self.mod == "64-QAM":
                mapper = mokka.mapping.torch.QAMConstellationMapper(6)
                constell = mapper.get_constellation().squeeze()
            self.num_lev = self.amp_levels.shape[0]
            self.P_tensor = torch.tensor(self.P, dtype=torch.float32)

            # # initialize net (butterfly FIR)

            # # add h_est as parameter
            # optimizer.add_param_group({"params": h_est})

            self.demapper = mokka.mapping.torch.ClassicalDemapper(  # IQDemapper(
                noise_sigma=torch.tensor(
                    0.1
                ),  # torch.sqrt(var[0]), #torch.tensor(0.3, requires_grad=True),  #var[0],  # 0.05
                # amp_levels=amp_levels,
                constellation=self.amp_levels,
                bitwise=False,
                optimize=False,
                p_symbols=self.P_tensor,
            )

            self.eqVAE = VAE_LE_DP(
                self.M,
                self.M,
                self.demapper,
                self.sps,
                block_size=self.batch_len,
                lr=self.lr_optim,
                requires_q=True,
                IQ_separate=True,
            )
            self.eqVAE.reset()

            self.SER_valid = torch.empty(4, self.num_frames, dtype=torch.float32)
            self.BMI = torch.zeros(self.num_frames, dtype=torch.float32)
            self.Var_est = torch.empty(self.pol, self.num_frames, dtype=torch.float32)

            self.minibatch = torch.empty(self.pol, 2, self.batch_len * self.sps, dtype=torch.float32)

            self.m_max = self.N_frame_max // self.batch_len
            self.N_frame = self.m_max * self.batch_len
            self.N_cut = 10  # number of symbols cut off to prevent edge effects of convolution

            (
                self.rx_tensor_full,
                self.data_tensor_full,
                self.sigma_n,
                self.temp_labels_full,
            ) = generate_data_shaping(
                self.num_frames * self.N_frame,
                self.amps,
                self.SNR,
                self.h_channel,
                self.P,
                self.pol,
                self.symb_rate,
                self.sps,
                self.tau_cd,
                self.tau_pmd,
                self.phiIQ,
                self.theta,
            )
            self.N_lrhalf = 10
        if self.current_frame % self.N_lrhalf == 0 and self.current_frame != 0:  # learning rate scheduler
            self.lr_optim *= 0.5
            self.eqVAE.update_lr = self.lr_optim

        with torch.set_grad_enabled(True):
            rx_tensor, data_tensor, temp_labels = (
                self.rx_tensor_full[
                    :, :, self.sps * self.current_frame * self.N_frame : self.sps * (self.current_frame + 1) * self.N_frame
                ],
                self.data_tensor_full[:, :, self.current_frame * self.N_frame : (self.current_frame + 1) * self.N_frame],
                self.temp_labels_full[:, self.current_frame * self.N_frame : (self.current_frame + 1) * self.N_frame],
            )
            bit_labels = self.bit_arr[temp_labels, :]
            self.theta += self.theta_diff  # update theta per frame

            out_train = torch.empty(
                self.pol,
                2 * self.num_lev,
                self.N_frame,
                dtype=torch.float32,
                requires_grad=False,
            )
            out_const = torch.empty(
                self.pol, 2, self.N_frame - self.batch_len, dtype=torch.float32, requires_grad=False
            )
            var_est = torch.empty(self.pol, self.m_max, dtype=torch.float32, requires_grad=False)

            # print(eq.butterfly_backward.taps)
            out, out_q = self.eqVAE(torch.complex(rx_tensor[:, 0, :], rx_tensor[:, 1, :]))

        out_const[:, 0, :], out_const[:, 1, :] = (
            out.real.detach().clone(),
            out.imag.detach().clone(),
        )

        out_train = out_q.permute(0, 2, 1)

        if self.current_frame == 0:
            self.out_full = out_const
        else:
            self.out_full = torch.cat((self.out_full, out_const), dim=2)

        temp_data_tensor = data_tensor[:, :, :-self.batch_len]

        shift, r = find_shift(
            out_train, temp_data_tensor, 21, self.amp_levels, self.pol
        )  # find correlation within 21 symbols

        out_train[0, :, :], out_train[1, :, :] = out_train[0, :, :].roll(
            int(-shift[0]), -1
        ), out_train[1, :, :].roll(
            int(-shift[1]), -1
        )  # compensate time shift (in multiple symb.)
        out_train = out_train.roll(r, 0)  # compensate pol. shift

        temp_out_train = out_train

        self.SER_valid[2:, self.current_frame], ind_IQ, ind_phase = SER_IQflip(
            temp_out_train[:, :, 11 : -11 - torch.max(torch.abs(shift))],
            temp_data_tensor[:, :, 11 : -11 - torch.max(torch.abs(shift))],
        )

        temp_bit_labels = bit_labels[:, :-self.batch_len, :]
        log_app = get_logAPPs(
            temp_out_train[:, :, 11 : -11 - torch.max(torch.abs(shift))],
            indIQ=ind_IQ,
            ind_phase=ind_phase,
        )
        self.BMI[self.current_frame] = bmi(
            log_app,
            temp_bit_labels[:, 11 : -11 - torch.max(torch.abs(shift)), :].reshape(
                2, -1
            ),
            self.H_P,
        )

        shift, r = find_shift_symb_full(
            out_const, temp_data_tensor, 21
        )  # find correlation within 21 symbols
        out_const[0, :, :], out_const[1, :, :] = out_const[0, :, :].roll(
            int(-shift[0]), -1
        ), out_const[1, :, :].roll(
            int(-shift[1]), -1
        )  # compensate time shift (in multiple symb.)
        out_const = out_const.roll(r, 0)  # compensate pol. shift
        temp_out_const = out_const

        self.SER_valid[:2, self.current_frame] = SER_constell_shaping(
            temp_out_const[:, :, 11 : -11 - torch.max(torch.abs(shift))]
            .detach()
            .clone(),
            temp_data_tensor[:, :, 11 : -11 - torch.max(torch.abs(shift))],
            self.amp_levels,
            self.nu_sc,
            self.var,
        )

        results = {
            "SER": self.SER_valid[:2, self.current_frame].detach().clone().cpu(),
            "bmi": self.BMI[self.current_frame].detach().clone().cpu(),
            "rx_signal_posteq": out.detach().clone().cpu(),
        }
        self.current_frame += 1
        return results




# Adapted functions from shared funcs

def init(channel, mod, nu, sps, M_est, SNR):
    if channel == "h1":  # h_1 in Caciularu et al.
        h_channel_orig = np.array(
            [
                0.0545 + 1j * 0.05,
                0.2823 - 1j * 0.11971,
                -0.7676 + 1j * 0.2788,
                -0.0641 - 1j * 0.0576,
                0.0466 - 1j * 0.02275,
            ]
        ).astype(np.complex64)
    elif channel == "h2":  # h_1 in Caciularu et al.
        h_channel_orig = np.array(
            [
                0.0545 + 1j * 0.0165,
                -1.3449 - 1j * 0.4523,
                1.0067 + 1j * 1.1524,
                0.3476 + 1j * 0.3153,
            ]
        ).astype(np.complex64)
    elif channel == "h0":  # only optical channel model, no further IR
        h_channel_orig = np.array([1]).astype(np.complex64)

    h_channel = np.zeros((sps * (h_channel_orig.shape[-1] - 1) + 1), dtype=np.complex64)
    h_channel[0::sps] = h_channel_orig  # upsampling channel IR by inserting zeros
    h_channel /= np.linalg.norm(h_channel)  # Normalization of the channel

    constellations = {
        "4-QAM": np.array([-1, -1, 1, 1]) + 1j * np.array([-1, 1, -1, 1]),
        "16-QAM": np.array([-3, -3, -3, -3, -1, -1, -1, -1, 1, 1, 1, 1, 3, 3, 3, 3])
        + 1j * np.array([-3, -1, 1, 3, -3, -1, 1, 3, -3, -1, 1, 3, -3, -1, 1, 3]),
        "64-QAM": np.array(
            [
                -7,
                -7,
                -7,
                -7,
                -7,
                -7,
                -7,
                -7,
                -5,
                -5,
                -5,
                -5,
                -5,
                -5,
                -5,
                -5,
                -3,
                -3,
                -3,
                -3,
                -3,
                -3,
                -3,
                -3,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                7,
                7,
                7,
                7,
                7,
                7,
                7,
                7,
            ]
        )
        + 1j
        * np.array(
            [
                -7,
                -5,
                -3,
                -1,
                1,
                3,
                5,
                7,
                -7,
                -5,
                -3,
                -1,
                1,
                3,
                5,
                7,
                -7,
                -5,
                -3,
                -1,
                1,
                3,
                5,
                7,
                -7,
                -5,
                -3,
                -1,
                1,
                3,
                5,
                7,
                -7,
                -5,
                -3,
                -1,
                1,
                3,
                5,
                7,
                -7,
                -5,
                -3,
                -1,
                1,
                3,
                5,
                7,
                -7,
                -5,
                -3,
                -1,
                1,
                3,
                5,
                7,
                -7,
                -5,
                -3,
                -1,
                1,
                3,
                5,
                7,
            ]
        ),
    }

    assert mod == "64-QAM"
    Gray_map = np.array(
        [
            0,
            1,
            3,
            2,
            6,
            7,
            5,
            4,
            8,
            9,
            11,
            10,
            14,
            15,
            13,
            12,
            24,
            25,
            27,
            26,
            30,
            31,
            29,
            28,
            16,
            17,
            19,
            18,
            22,
            23,
            21,
            20,
            48,
            49,
            51,
            50,
            54,
            55,
            53,
            52,
            56,
            57,
            59,
            58,
            62,
            63,
            61,
            60,
            40,
            41,
            43,
            42,
            46,
            47,
            45,
            44,
            32,
            33,
            35,
            34,
            38,
            39,
            37,
            36,
        ],
        dtype=np.uint8,
    )
    bit_arr = torch.tensor(np.unpackbits(Gray_map).reshape(64, 8), dtype=torch.int8)
    bit_arr = bit_arr[:, 8 - 6 :]

    pol = 2  # number of channels (polarizations)

    constellation = constellations[mod] / np.sqrt(
        np.mean(np.abs(constellations[mod]) ** 2)
    )  # normalize modulation format

    amp_levels = (
        constellation.real
    )  # ASK levels (poitive and negative amplitude levels)
    num_lev = int(np.sqrt(len(amp_levels)))  # number of ASK levels
    amps = amp_levels[::num_lev]  # amplitude levels
    amp_levels = torch.tensor(amps, dtype=torch.float32)
    sc = np.min(np.abs(amps))  # scaling factor for having lowest level equal 1
    nu_sc = nu / sc**2  # re-scaled shaping factor

    P = np.exp(-nu * np.abs(amps / sc) ** 2)
    P = P / np.sum(P)  # pmf of the amlitude levels

    shape_mat = np.zeros((num_lev, num_lev))
    for i in range(num_lev):
        shape_mat[i, :] = P
    P_mat = (shape_mat * shape_mat.T) / np.sum(
        shape_mat * shape_mat.T
    )  # matrix with the corresponding probabilities for each constellation point
    H_P = -np.sum(np.log2(P_mat) * P_mat)  # entropy of the modulation format
    pow_mean = np.sum(
        P_mat.reshape(-1) * np.abs(constellation) ** 2
    )  # mean power of the constellation

    kurtosis = (
        np.sum(P_mat.reshape(-1) * np.abs(constellation) ** 4) / pow_mean
    )  # E{|s|**4}/E{|s|**2}

    var = torch.full(
        (2,), pow_mean / 10 ** (SNR / 10) / 2, dtype=torch.float32
    )  # noise variance for the soft demapper

    h_est = np.zeros([pol, pol, 2, M_est])  # initialize estimated impulse response
    h_est[0, 0, 0, M_est // 2 + 1], h_est[1, 1, 0, M_est // 2 + 1] = (
        1,
        1,
    )  # 0.5, 0.5     # Dirac initialization
    h_est = torch.tensor(h_est, requires_grad=True, dtype=torch.float32)

    return (
        h_est,
        h_channel,
        P,
        amp_levels,
        amps,
        bit_arr,
        pol,
        nu_sc,
        var,
        pow_mean,
        kurtosis,
        H_P,
    )  # pow_mean


def generate_data_shaping(
    N, amps, SNR, h_channel, P, pol, symb_rate, sps, tau_cd, tau_pmd, phiIQ, theta
):
    T = 8  # length of pulse-shaping filter in symbols
    beta = 0.1  # roll-off factor

    M = len(h_channel)  # number of channel taps
    m_amps = amps.shape[-1]

    N_conv = N + len(h_channel) + 4 * T
    tx_up = np.zeros((pol, sps * (N_conv - 1) + 1), dtype=np.complex64)
    rx_sig = np.zeros((pol, sps * N), dtype=np.complex64)

    rng = np.random.default_rng()
    labels = rng.choice(range(m_amps), (pol * 2, N_conv), p=P)
    data = amps[labels]
    # data = rng.choice(amps, (pol*2,N_conv), p=P)    # draw random amplitude level from corresponding pmf P
    tx_up[:, ::sps] = (
        data[0::pol, :] + 1j * data[1::pol, :]
    )  # sps-upsampled signal by zero-insertion

    h_pulse = rrcfir(T, sps, beta)
    temp = simulate_channel(tx_up, h_pulse, h_channel)
    temp = simulate_dispersion(temp, symb_rate, sps, tau_cd, tau_pmd, phiIQ, theta)

    sigma_n = np.sqrt(
        np.mean(np.abs(temp) ** 2) * sps / 2 / 10 ** (SNR / 10)
    )  # var/2 due to I/Q, *sps due to oversampling with zeros
    temp += sigma_n * (
        np.random.randn(*temp.shape) + 1j * np.random.randn(*temp.shape)
    )  # Standard-normal distribution with exp(1/2*x**2)

    rx_sig = temp

    rx_tensor = (
        torch.from_numpy(
            np.asarray([np.real(rx_sig[:, : sps * N]), np.imag(rx_sig[:, : sps * N])])
        )
        .permute(1, 0, 2)
        .to(torch.float32)
    )
    data_tensor = (
        torch.from_numpy(
            np.asarray(
                [
                    data[0::pol, (T + M - 1) : (N + T + M - 1)],
                    data[1::pol, (T + M - 1) : (N + T + M - 1)],
                ]
            )
        )
        .permute(1, 0, 2)
        .to(torch.float16)
    )

    labels_tensor = (
        torch.from_numpy(
            np.asarray(
                [
                    labels[0::pol, (T + M - 1) : (N + T + M - 1)],
                    labels[1::pol, (T + M - 1) : (N + T + M - 1)],
                ]
            )
        )
        .permute(1, 0, 2)
        .to(torch.long)
    )

    symb_labels = labels_tensor[:, 0, :] * m_amps + labels_tensor[:, 1, :]

    return rx_tensor, data_tensor, sigma_n, symb_labels


def rrcfir(T, sps, beta):  # root-raised-cosine filter
    # T = 6 # pulse duration in symbols
    # sps = 2 # oversampling factor in samples per symbol
    # beta = 0.1 # roll-off factor
    t = np.arange(-T * sps / 2, T * sps / 2, 1 / sps, dtype=np.float32)
    ind_zero, ind_4beta = (t == 0), (np.abs(t) == 1 / 4 / beta)
    t[ind_zero], t[ind_4beta] = 1e-5, 1e-5
    h = (
        np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
    ) / (np.pi * t * (1 - (4 * beta * t) ** 2))
    h[ind_4beta] = (
        beta
        / np.sqrt(2)
        * (
            (1 + 2 / np.pi) * np.sin(np.pi / 4 / beta)
            + (1 - 2 / np.pi) * np.cos(np.pi / 4 / beta)
        )
    )
    h[ind_zero] = 1 + beta * (4 / np.pi - 1)
    h = h / np.linalg.norm(h)  # Normalisation of the pulseforming filter
    return h


def simulate_dispersion(rx, symb_rate, sps, tau_cd, tau_pmd, phiIQ, theta):
    # simulate residual CD, PMD, pol. rot and IQ-shift in f-domain
    rx_fft = np.fft.fft(rx, axis=1)
    freq = np.fft.fftfreq(rx.shape[1], 1 / symb_rate / sps)
    exp_cd, exp_pmd = np.exp(1j * 2 * (np.pi * freq) ** 2 * tau_cd), np.exp(
        1j * np.pi * tau_pmd * freq
    )
    rho = (
        0 * np.pi
    )  # 0.1*np.pi # input+output pol shift? # np.random.uniform(0,2*np.pi) # shift of PSP to reference
    cos_rho, sin_rho = np.cos(rho), np.sin(rho)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)  #
    exp_phiIQ = np.exp(-1j * phiIQ)

    # simulate pol. rotation and PMD with rotationary matrix
    # R = np.asarray([[cos_theta*exp_phiIQ[0], sin_theta*exp_phiIQ[0]], [-sin_theta*exp_phiIQ[1], cos_theta*exp_phiIQ[1]]])
    # R_T = np.asarray([[cos_theta*exp_phiIQ[0], -sin_theta*exp_phiIQ[0]], [sin_theta*exp_phiIQ[1], cos_theta*exp_phiIQ[1]]])
    R_1 = np.asarray([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
    R_2_IQ = np.asarray(
        [
            [cos_rho * exp_phiIQ[0], sin_rho * exp_phiIQ[0]],
            [-sin_rho * exp_phiIQ[1], cos_rho * exp_phiIQ[1]],
        ]
    )
    # Diag_pmd = np.asarray([[exp_pmd, 0], [0, 1/exp_pmd]])
    Diag_pmd = np.asarray(
        [[exp_pmd, np.zeros_like(exp_pmd)], [np.zeros_like(exp_pmd), 1 / exp_pmd]]
    )
    # H = R_T @ Diag_pmd #@ R
    H = R_1 @ Diag_pmd.transpose(2, 0, 1) @ R_2_IQ

    RX_fft = np.zeros((2, rx.shape[1]), dtype=np.complex128)
    RX_fft[0, :], RX_fft[1, :] = (
        H[:, 0, 0] * rx_fft[0, :] + H[:, 0, 1] * rx_fft[1, :]
    ) * exp_cd, (H[:, 1, 0] * rx_fft[0, :] + H[:, 1, 1] * rx_fft[1, :]) * exp_cd
    # RX_fft[0,:], RX_fft[1,:] = (H[0,0]*rx_fft[0,:] + H[0,1]*rx_fft[1,:])*exp_cd, (H[1,0]*rx_fft[0,:] + H[1,1]*rx_fft[1,:])*exp_cd
    return np.complex64(np.fft.ifft(RX_fft, axis=1))  # return signal in t-domain


def simulate_channel(tx_up, h_pulse, h_channel):
    pol = tx_up.shape[0]
    rx_sig = np.zeros(
        (pol, tx_up.shape[1] - h_pulse.shape[0] - h_channel.shape[0] + 2),
        dtype=np.complex64,
    )

    for i in range(tx_up.shape[0]):  # num. of pol.
        temp = np.convolve(
            tx_up[i, :], h_pulse, mode="valid"
        )  # convolve with pulse shaping
        rx_sig[i, :] = np.convolve(
            temp, h_channel, mode="valid"
        )  # convolve with (additional) channel IR
    return rx_sig


def find_shift(q, tx, N_shift, amp_levels, pol):
    # find shiftings in both polarization and time by correlation with expectation of x^I with respect to q
    corr_max = torch.empty(2, 2, 2, device=q.device, dtype=torch.float32)
    num_lev = q.shape[1] // 2
    corr_ind = torch.empty_like(corr_max)
    len_corr = q.shape[-1]
    amp_mat = amp_levels.repeat(pol, len_corr, 1).transpose(1, 2)
    E = torch.sum(
        amp_mat * q[:, :num_lev, :len_corr], dim=1
    )  # calculate expectation E_q(x^I) of in-phase component

    # correlate with (both polarizations) and shifted versions in time --> find max. correlation
    E_mat = torch.empty(2, len_corr, N_shift, device=q.device, dtype=torch.float32)
    for i in range(N_shift):
        E_mat[:, :, i] = torch.roll(E, i - N_shift // 2, -1)
    corr_max[0, :, :], corr_ind[0, :, :] = torch.max(
        torch.abs(tx[:, 0, :len_corr].float() @ E_mat), dim=-1
    )
    corr_max[1, :, :], corr_ind[1, :, :] = torch.max(
        torch.abs(tx[:, 1, :len_corr].float() @ E_mat), dim=-1
    )
    corr_max, ind_max = torch.max(corr_max, dim=0)
    # corr_ind = corr_ind[ind_max]

    ind_XY = torch.zeros(2, device=q.device, dtype=torch.int16)
    ind_YX = torch.zeros_like(ind_XY)
    ind_XY[0] = corr_ind[ind_max[0, 0], 0, 0]
    ind_XY[1] = corr_ind[ind_max[1, 1], 1, 1]
    ind_YX[0] = corr_ind[ind_max[0, 1], 0, 1]
    ind_YX[1] = corr_ind[ind_max[1, 0], 1, 0]

    if (corr_max[0, 0] + corr_max[1, 1]) >= (corr_max[0, 1] + corr_max[1, 0]):
        return N_shift // 2 - ind_XY, 0
    else:
        return N_shift // 2 - ind_YX, 1


def SER_IQflip(q, tx):
    # estimate symbol error rate from estimated a posterioris q
    device = q.device
    num_lev = q.shape[1] // 2
    dec = torch.empty_like(tx, device=device, dtype=torch.int16)
    data = torch.empty_like(tx, device=device, dtype=torch.int16)
    data_IQinv = torch.empty_like(data)
    SER = torch.ones(2, 2, 4, device=device, dtype=torch.float32)

    scale = (num_lev - 1) / 2
    data = torch.round(scale * tx.float() + scale)  # decode TX
    data_IQinv[:, 0, :], data_IQinv[:, 1, :] = data[:, 0, :], -(
        data[:, 1, :] - scale * 2
    )  # compensate potential IQ flip
    ### zero phase-shift
    dec[:, 0, :], dec[:, 1, :] = torch.argmax(q[:, :num_lev, :], dim=1), torch.argmax(
        q[:, num_lev:, :], dim=1
    )  # hard decision on max(q)
    SER[0, :, 0] = torch.mean(
        ((data - dec).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32
    )
    SER[1, :, 0] = torch.mean(
        ((data_IQinv - dec).bool().any(dim=1)).to(torch.float),
        dim=-1,
        dtype=torch.float32,
    )

    ### pi phase-shift
    dec_pi = -(dec - scale * 2)
    SER[0, :, 1] = torch.mean(
        ((data - dec_pi).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32
    )
    SER[1, :, 1] = torch.mean(
        ((data_IQinv - dec_pi).bool().any(dim=1)).to(torch.float),
        dim=-1,
        dtype=torch.float32,
    )

    ### pi/4 phase-shift
    dec_pi4 = torch.empty_like(dec)
    dec_pi4[:, 0, :], dec_pi4[:, 1, :] = -(dec[:, 1, :] - scale * 2), dec[:, 0, :]
    SER[0, :, 2] = torch.mean(
        ((data - dec_pi4).bool().any(dim=1)).to(torch.float),
        dim=-1,
        dtype=torch.float32,
    )
    SER[1, :, 2] = torch.mean(
        ((data_IQinv - dec_pi4).bool().any(dim=1)).to(torch.float),
        dim=-1,
        dtype=torch.float32,
    )

    ### 3pi/4 phase-shift
    dec_3pi4 = -(dec_pi4 - scale * 2)
    SER[0, :, 3] = torch.mean(
        ((data - dec_3pi4).bool().any(dim=1)).to(torch.float),
        dim=-1,
        dtype=torch.float32,
    )
    SER[1, :, 3] = torch.mean(
        ((data_IQinv - dec_3pi4).bool().any(dim=1)).to(torch.float),
        dim=-1,
        dtype=torch.float32,
    )

    # SER_out = torch.amin(SER, dim=(0,-1))   # choose minimum estimation per polarization
    SER_out_temp, ind_IQ = torch.min(SER, dim=0)
    SER_out, ind_phase = torch.min(SER_out_temp, dim=-1)

    ind_IQ_out = torch.empty_like(ind_phase)
    for i in range(tx.shape[0]):
        ind_IQ_out[i] = ind_IQ[i, ind_phase[i]]
    return SER_out, ind_IQ_out, ind_phase  # /num_bit


def get_logAPPs(q, indIQ=0, ind_phase=0):
    """
    Computes the logarithm of the a-posterioris for each symbol

    :param q: a-posterioris per I and Q
    :returns symb_apps
    """
    m_root = q.shape[1] // 2
    for p in range(q.shape[0]):
        if indIQ[p] != 0:
            q_temp = torch.empty(
                q.shape[1], q.shape[-1], dtype=q.dtype, device=q.device
            )
            q_temp[:m_root, :], q_temp[m_root:, :] = q[p, :m_root, :], q[
                p, m_root:, :
            ].flip(0)
            q[p, :, :] = q_temp
        if ind_phase[p] != 0:
            q_temp = torch.empty(
                q.shape[1], q.shape[-1], dtype=q.dtype, device=q.device
            )
            if ind_phase[p] == 1:
                q_temp[:m_root, :], q_temp[m_root:, :] = q[p, :m_root, :].flip(0), q[
                    p, m_root:, :
                ].flip(0)
            if ind_phase[p] == 2:
                q_temp[:m_root, :], q_temp[m_root:, :] = (
                    q[p, m_root:, :].flip(0),
                    q[p, :m_root, :],
                )
            if ind_phase[p] == 3:
                q_temp[:m_root, :], q_temp[m_root:, :] = q[p, m_root:, :], q[
                    p, :m_root, :
                ].flip(0)
            q[p, :, :] = q_temp

    symb_apps = torch.zeros(
        q.shape[0], q.shape[-1], m_root**2, device=q.device, dtype=torch.float32
    )
    for i in range(m_root):
        for j in range(m_root):
            symb_apps[:, :, i * m_root + j] = q[:, i, :] * q[:, m_root + j, :]
    symb_apps += 1e-17
    symb_apps /= torch.sum(symb_apps, dim=2, keepdim=True)
    return torch.log(symb_apps)


def bmi(log_app, label_bits, H_P, a=0.0, b=1.0, tol=1e-5):
    """
    Computes the bitwise mutual information (BMI).

    :param log_app: Logarithmic APP estimations of the symbol detector.
    :param label_bits: The actually sent bits.
    :returns BMI (scalar value)
    """
    hefu_cl = hefu_class(log_app.device)
    # Apply bit metric decoder to log APPs of symbols to get bit-wise LLRs.
    assert log_app.shape[-2] == label_bits.shape[-1] / hefu_cl.m
    assert (
        log_app.shape[-1] == hefu_cl.M
    )  # The last shape should be the log-probabilities of the symbols.
    llrs = hefu_cl.bit_metric_decoder(log_app)
    # return constellation.m * (1 - t.mean(t.log2(1+t.exp((2*label_bits-1) * llrs))))
    # return hefu_cl.m * (1 - torch.mean(1/np.log(2) * (torch.clamp((2*label_bits-1) * llrs, 0) + torch.log(1+torch.exp(-torch.abs((2*label_bits-1) * llrs))))))
    # return  bmi_llr(llrs, label_bits, hefu_cl, H_P)

    gr = torch.tensor((np.sqrt(5) + 1) / 2, device=llrs.device, dtype=torch.float32)
    a = torch.tensor(a, device=llrs.device, dtype=torch.float32)
    b = torch.tensor(b, device=llrs.device, dtype=torch.float32)

    # golden search
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while torch.abs(b - a) > tol:
        if bmi_llr(c * llrs, label_bits, hefu_cl, H_P) > bmi_llr(
            d * llrs, label_bits, hefu_cl, H_P
        ):
            b = d
        else:
            a = c
        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return bmi_llr(((b + a) / 2) * llrs, label_bits, hefu_cl, H_P)


def bmi_llr(llrs, label_bits, hefu_cl, H_P):
    """
    Computes the bitwise mutual information (BMI).

    :param llrs: LLR estimations of the symbol detector.
    :param label_bits: The actually sent bits.
    :returns BMI (scalar value)
    """
    return H_P - hefu_cl.m * torch.mean(
        1
        / np.log(2)
        * (
            torch.clamp((2 * label_bits - 1) * llrs, 0)
            + torch.log(1 + torch.exp(-torch.abs((2 * label_bits - 1) * llrs)))
        )
    )


def find_shift_symb_full(rx, tx, N_shift):
    # find shiftings in both polarization and time by correlation with the constellation output's in-phase component x^I
    corr_max = torch.empty(2, 2, 2, device=rx.device, dtype=torch.float32)
    corr_ind = torch.empty_like(corr_max)
    len_corr = rx.shape[-1]  # torch.max(q.shape[-1],1000)
    E = rx[:, 0, :len_corr]

    # correlate with (both polarizations) and shifted versions in time --> find max. correlation
    E_mat = torch.empty(2, len_corr, N_shift, device=rx.device, dtype=torch.float32)
    for i in range(N_shift):
        E_mat[:, :, i] = torch.roll(E, i - N_shift // 2, -1)
    corr_max[0, :, :], corr_ind[0, :, :] = torch.max(
        torch.abs(tx[:, 0, :len_corr].float() @ E_mat), dim=-1
    )
    corr_max[1, :, :], corr_ind[1, :, :] = torch.max(
        torch.abs(tx[:, 1, :len_corr].float() @ E_mat), dim=-1
    )
    corr_max, ind_max = torch.max(corr_max, dim=0)

    ind_XY = torch.zeros(2, device=rx.device, dtype=torch.int16)
    ind_YX = torch.zeros_like(ind_XY)
    ind_XY[0] = corr_ind[ind_max[0, 0], 0, 0]
    ind_XY[1] = corr_ind[ind_max[1, 1], 1, 1]
    ind_YX[0] = corr_ind[ind_max[0, 1], 0, 1]
    ind_YX[1] = corr_ind[ind_max[1, 0], 1, 0]

    if (corr_max[0, 0] + corr_max[1, 1]) >= (corr_max[0, 1] + corr_max[1, 0]):
        return N_shift // 2 - ind_XY, 0
    else:
        return N_shift // 2 - ind_YX, 1


def SER_constell_shaping(rx, tx, amp_levels, nu_sc, var):
    # estimate symbol error rate from output constellation by considering PCS
    device = rx.device
    num_lev = amp_levels.shape[0]
    data = torch.empty_like(tx, device=device, dtype=torch.int32)
    data_IQinv = torch.empty_like(data)
    SER = torch.ones(2, 2, 4, device=device, dtype=torch.float32)

    # calculate decision boundaries based on PCS
    d_vec = (1 + 2 * nu_sc * var[0]) * (amp_levels[:-1] + amp_levels[1:]) / 2
    d_vec0 = torch.cat(((-Inf * torch.ones(1, device=device)), d_vec), dim=0)
    d_vec1 = torch.cat((d_vec, Inf * torch.ones(1, device=device)))

    scale = (num_lev - 1) / 2
    data = torch.round(scale * tx.float() + scale).to(torch.int32)  # decode TX
    data_IQinv[:, 0, :], data_IQinv[:, 1, :] = data[:, 0, :], -(
        data[:, 1, :] - scale * 2
    )  # compensate potential IQ flip

    rx *= torch.mean(
        torch.sqrt(tx[:, 0, :].float() ** 2 + tx[:, 1, :].float() ** 2)
    ) / torch.mean(
        torch.sqrt(rx[:, 0, :] ** 2 + rx[:, 1, :] ** 2)
    )  # normalize constellation output

    ### zero phase-shift  torch.sqrt(2*torch.mean(rx[0,:N*sps:sps]**2))
    SER[0, :, 0] = dec_on_bound(rx, data, d_vec0, d_vec1)
    SER[1, :, 0] = dec_on_bound(rx, data_IQinv, d_vec0, d_vec1)

    ### pi phase-shift
    rx_pi = -(rx).detach().clone()
    SER[0, :, 1] = dec_on_bound(rx_pi, data, d_vec0, d_vec1)
    SER[1, :, 1] = dec_on_bound(rx_pi, data_IQinv, d_vec0, d_vec1)

    ### pi/4 phase-shift
    rx_pi4 = torch.empty_like(rx)
    rx_pi4[:, 0, :], rx_pi4[:, 1, :] = -(rx[:, 1, :]).detach().clone(), rx[:, 0, :]
    SER[0, :, 2] = dec_on_bound(rx_pi4, data, d_vec0, d_vec1)
    SER[1, :, 2] = dec_on_bound(rx_pi4, data_IQinv, d_vec0, d_vec1)

    ### 3pi/4 phase-shift
    rx_3pi4 = -(rx_pi4).detach().clone()
    SER[0, :, 3] = dec_on_bound(rx_3pi4, data, d_vec0, d_vec1)
    SER[1, :, 3] = dec_on_bound(rx_3pi4, data_IQinv, d_vec0, d_vec1)

    SER_out = torch.amin(SER, dim=(0, -1))  # choose minimum estimation per polarization
    return SER_out


def dec_on_bound(rx, tx_int, d_vec0, d_vec1):
    # hard decision based on the decision boundaries d_vec0 (lower) and d_vec1 (upper)
    SER = torch.zeros(rx.shape[0], dtype=torch.float32, device=rx.device)

    xI0 = d_vec0.index_select(dim=0, index=tx_int[0, 0, :])
    xI1 = d_vec1.index_select(dim=0, index=tx_int[0, 0, :])
    corr_xI = torch.bitwise_and((xI0 <= rx[0, 0, :]), (rx[0, 0, :] < xI1))
    xQ0 = d_vec0.index_select(dim=0, index=tx_int[0, 1, :])
    xQ1 = d_vec1.index_select(dim=0, index=tx_int[0, 1, :])
    corr_xQ = torch.bitwise_and((xQ0 <= rx[0, 1, :]), (rx[0, 1, :] < xQ1))

    yI0 = d_vec0.index_select(dim=0, index=tx_int[1, 0, :])
    yI1 = d_vec1.index_select(dim=0, index=tx_int[1, 0, :])
    corr_yI = torch.bitwise_and((yI0 <= rx[1, 0, :]), (rx[1, 0, :] < yI1))
    yQ0 = d_vec0.index_select(dim=0, index=tx_int[1, 1, :])
    yQ1 = d_vec1.index_select(dim=0, index=tx_int[1, 1, :])
    corr_yQ = torch.bitwise_and((yQ0 <= rx[1, 1, :]), (rx[1, 1, :] < yQ1))

    ex, ey = ~(torch.bitwise_and(corr_xI, corr_xQ)), ~(
        torch.bitwise_and(corr_yI, corr_yQ)
    )  # no error only if both I or Q are correct
    SER[0], SER[1] = (
        torch.sum(ex) / ex.nelement(),
        torch.sum(ey) / ey.nelement(),
    )  # SER = numb. of errors/ num of symbols
    return SER


"""
Constellation class.
"""


class hefu_class:
    """
    Class which provides some functions, applied to an arbitrary complex constellation,
    given in mapping.
    """

    def __init__(self, device):
        """
        :param mapping: t.Tensor which contains the constellation symbols, sorted according
            to their binary representation (MSB left).
        """
        qam64_mapping_1 = t.tensor(
            np.array(
                [
                    -7,
                    -7,
                    -7,
                    -7,
                    -7,
                    -7,
                    -7,
                    -7,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -3,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                ]
            )
            + 1j
            * np.array(
                [
                    -7,
                    -5,
                    -3,
                    -1,
                    1,
                    3,
                    5,
                    7,
                    -7,
                    -5,
                    -3,
                    -1,
                    1,
                    3,
                    5,
                    7,
                    -7,
                    -5,
                    -3,
                    -1,
                    1,
                    3,
                    5,
                    7,
                    -7,
                    -5,
                    -3,
                    -1,
                    1,
                    3,
                    5,
                    7,
                    -7,
                    -5,
                    -3,
                    -1,
                    1,
                    3,
                    5,
                    7,
                    -7,
                    -5,
                    -3,
                    -1,
                    1,
                    3,
                    5,
                    7,
                    -7,
                    -5,
                    -3,
                    -1,
                    1,
                    3,
                    5,
                    7,
                    -7,
                    -5,
                    -3,
                    -1,
                    1,
                    3,
                    5,
                    7,
                ]
            ),
            dtype=t.cfloat,
        )

        Gray_map = t.tensor(
            [
                0,
                1,
                3,
                2,
                6,
                7,
                5,
                4,
                8,
                9,
                11,
                10,
                14,
                15,
                13,
                12,
                24,
                25,
                27,
                26,
                30,
                31,
                29,
                28,
                16,
                17,
                19,
                18,
                22,
                23,
                21,
                20,
                48,
                49,
                51,
                50,
                54,
                55,
                53,
                52,
                56,
                57,
                59,
                58,
                62,
                63,
                61,
                60,
                40,
                41,
                43,
                42,
                46,
                47,
                45,
                44,
                32,
                33,
                35,
                34,
                38,
                39,
                37,
                36,
            ],
            dtype=t.long,
        )
        qam64_mapping = qam64_mapping_1[Gray_map]
        bit_arr = t.tensor(np.unpackbits(Gray_map.to(t.uint8)).reshape(64, 8))
        bit_arr = bit_arr[:, 8 - 6 :]
        O = bit_arr.nonzero()  # A[bit_arr==0]
        Z = (1 - bit_arr).nonzero()
        S = t.zeros(6, 2, 32)
        for k in range(6):
            S[k, 1, :] = O[O[:, 1] == k, 0]  # ones
            S[k, 0, :] = Z[Z[:, 1] == k, 0]  # zeros

        assert len(qam64_mapping.shape) == 1  # mapping should be a 1-dim tensor
        self.mapping = qam64_mapping.to(device)

        self.M = t.numel(self.mapping)  # Number of constellation symbols.
        self.m = np.log2(self.M).astype(int)
        assert self.m == np.log2(self.M)  # Assert that log2(M) is integer
        self.mask = 2 ** t.arange(self.m - 1, -1, -1).to(device)

        self.sub_consts = S.to(dtype=t.int32, device=device)
        # self.sub_consts = t.stack([t.stack([t.arange(self.M).reshape(2**(i+1),-1)[::2].flatten(), t.arange(self.M).reshape(2**(i+1),-1)[1::2].flatten()]) for i in range(self.m)]).to(device)
        self.device = device

    def map(self, bits):
        """
        Maps a given bit_sequence to a sequence of constellation symbols.
        The length of the output sequence is len(bit_sequence) / m.
        The operation is applied to the last axis of bit_sequences.
        bit_sequence is allowed to have other dimensions (e.g. multiple sequences at once)
        as long as the last dimensions is the sequence.
        """
        # Assert that the length of the bit sequence is a multiple of m.
        in_shape = bits.shape
        assert in_shape[-1] / self.m == in_shape[-1] // self.m
        # reshape and convert bits to decimal and use decimal number as index for mapping
        return self.mapping[
            t.sum(self.mask * bits.reshape(in_shape[:-1] + (-1, self.m)), -1)
        ]

    def bit2symbol_idx(self, bits):
        """
        Returns the symbol number (sorted as in self.mapping) for an incoming sequence of bits.
        This "symbol number" can be used for one-hot representation, for example.
        The length of the output sequence is len(bit_sequence) / m.
        The operation is applied to the last axis of bit_sequences.
        bit_sequence is allowed to have other dimensions (e.g. multiple sequences at once)
        as long as the last dimensions is the sequence.
        """
        # Assert that the length of the bit sequence is a multiple of m.
        in_shape = bits.shape
        assert in_shape[-1] / self.m == in_shape[-1] // self.m
        # reshape and convert bits to decimal and use decimal number as index for mapping
        return t.sum(self.mask * bits.reshape(in_shape[:-1] + (-1, self.m)), -1)

    def demap(self, symbol_idxs):
        """
        Demaps a sequence of constellation symbols, given by their indices in self.mapping, to a sequence of bits.
        The length of the output sequence is len(symbols) * m.
        The operation is applied to the last axis of the input sequence.
        """
        # Assert that the length of the bit sequence is a multiple of m.
        in_shape = symbol_idxs.shape
        # reshape and convert symbol to bits and use decimal number as index for mapping
        return (
            symbol_idxs.unsqueeze(-1)
            .bitwise_and(self.mask)
            .ne(0)
            .view(symbol_idxs.shape[:-1] + (-1,))
            .float()
        )

    def nearest_neighbor(self, rx_syms):
        """
        Accepts a sequence of (possibly equalized) complex symbols.
        Each sample is hard decided to the constellation symbol, which is nearest (Euclidean distance).
        The output are the idxs of the constellation symbols.
        """
        # Compute distances to all possible symbols.
        distance = t.abs(self.mapping - rx_syms[..., None])
        hard_dec_idx = t.argmin(distance, dim=-1)
        return hard_dec_idx

    def bit_metric_decoder(self, symbol_apps):
        """
        Receives a sequence of symbol APPs. For each symbol, an M-dim tensor indicates the logarithmic
        probability for each of the M possible constellation symbols.
        The bit metric decoder calculates the bit LLRs for each of the m bits for each symbol.
        """
        assert (
            len(symbol_apps.shape) >= 2
        )  # second last: symbol sequence, last: M log APPs
        assert symbol_apps.shape[-1] == self.M

        # For each of the m bits, repartition the M APPs into two subsets regarding the respective bit.
        # The output vector has shape (..., m, 2, M/2).
        subset_probs = t.index_select(symbol_apps, -1, self.sub_consts.flatten()).view(
            symbol_apps.shape[:-1] + self.sub_consts.shape
        )
        # Sum up probabilities of all subsets. (exp to go from log to lin domain and log to go back to log domain)
        bitwise_apps = self.jacobian_sum(subset_probs, dim=-1)
        # LLR
        LLR = (bitwise_apps[..., 0] - bitwise_apps[..., 1]).flatten(start_dim=-2)
        assert symbol_apps.shape[:-2] == LLR.shape[:-1]
        assert symbol_apps.shape[-2] * self.m == LLR.shape[-1]
        assert not t.isinf(LLR).any()

        return LLR

    def jacobian_sum(self, msg, dim):
        """
        Computes ln(e^a_1 + e^a_2 + ... e^a_M) of a tensor with last dimension (a_1, a_2, ..., a_M)
        by applying the Jacobian algorithm recursively.
        """
        if msg.shape[dim] == 1:
            return msg.flatten(start_dim=-2)
        else:
            if dim == -1:
                tmp = self.pairwise_jacobian_sum(msg[..., 0], msg[..., 1])
                for i in range(2, msg.shape[-1]):
                    tmp = self.pairwise_jacobian_sum(tmp, msg[..., i])
            elif dim == -2:
                tmp = self.pairwise_jacobian_sum(msg[..., 0, :], msg[..., 1, :])
                for i in range(2, msg.shape[-1]):
                    tmp = self.pairwise_jacobian_sum(tmp, msg[..., i, :])
            return tmp

    def pairwise_jacobian_sum(self, msg1, msg2):
        """
        Computes ln(e^msg1 + e^msg2) = max(msg1,msg2) + ln(1+e^-|msg1-msg2|).
        """
        return t.maximum(msg1, msg2) + t.log(1 + t.exp(-t.abs(msg1 - msg2)))
