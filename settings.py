from enum import IntEnum

class ShapingType(IntEnum):
    Geometric =  0
    Probabilistic = 1
    Joint = 1

class ShapingChannel(IntEnum):
    AWGN = 0
    Wiener =  1
    Optical = 2

class Demapper(IntEnum):
    Neural = 0
    Gaussian = 1
    SepNeural = 2
    SepGaussian = 3

class CPE(IntEnum):
    NONE = 0
    BPS = 1
    VV = 2

class ShapingObjective(IntEnum):
    BMI = 0
    MI = 1
