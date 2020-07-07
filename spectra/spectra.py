
from enum import Enum
from typing import Union, Iterable, Callable

import math
import numpy as np                  # type: ignore
import numpy.linalg as la           # type: ignore
from numpy import ndarray
import matplotlib.pyplot as plt     # type: ignore


# Characteristic colors
UVI = 380     #: Ultra-violet
VIO = 420     #: Violet
BLU = 440     #: Blue
CYA = 490     #: Cyan
GRN = 510     #: Green
YEL = 580     #: Yellow
RED = 645     #: Red
RE2 = 700     #: Red
IRD = 780     #: Infra-red

# Intervals
UV = VIO - UVI
BC = CYA - BLU
CG = GRN - CYA
GY = YEL - GRN
YR = RED - YEL
IR = IRD - RE2


def wavelength_to_rgb(w: float, gamma: float = 1.) -> ndarray:
    """
    Approximate conversion of wavelength to RGB value

    http://lsrtools.1apps.com/wavetorgb/index.asp
    http://www.physics.sfasu.edu/astro/color/spectra.html

    :param w:  wavelength in nm
    :param gamma: monitor's gamma
    :return: RGB values between 0 and 1
    """

    if w <= BLU:
        c = (BLU - w) / 90, 0., 1.  # violet - blue
    elif w <= CYA:
        c = 0., (w - BLU) / BC, 1.  # blue - cyan
    elif w <= GRN:
        c = 0., 1., (GRN - w) / CG  # cyan - green
    elif w <= YEL:
        c = (w - GRN) / GY, 1., 0.  # green - yellow
    elif w <= RED:
        c = 1., (RED - w) / YR, 0.  # yellow - orange - red
    else:
        c = 1., 0., 0.              # red

    rgb = np.array(c)

    if w < VIO:
        rgb *= .3 + .7 * (w - UVI) / UV
    elif w > RE2:
        rgb *= .3 + .7 * (IRD - w) / IR

    return rgb.clip(0) ** gamma


# Wavelength to CIE XYZ from 380 to 780 np in steps of 5 nm.
CIE_XYZ_380_780 = np.array((
    (0.0014, 0.0000, 0.0065), (0.0022, 0.0001, 0.0105), (0.0042, 0.0001, 0.0201),
    (0.0076, 0.0002, 0.0362), (0.0143, 0.0004, 0.0679), (0.0232, 0.0006, 0.1102),
    (0.0435, 0.0012, 0.2074), (0.0776, 0.0022, 0.3713), (0.1344, 0.0040, 0.6456),
    (0.2148, 0.0073, 1.0391), (0.2839, 0.0116, 1.3856), (0.3285, 0.0168, 1.6230),
    (0.3483, 0.0230, 1.7471), (0.3481, 0.0298, 1.7826), (0.3362, 0.0380, 1.7721),
    (0.3187, 0.0480, 1.7441), (0.2908, 0.0600, 1.6692), (0.2511, 0.0739, 1.5281),
    (0.1954, 0.0910, 1.2876), (0.1421, 0.1126, 1.0419), (0.0956, 0.1390, 0.8130),
    (0.0580, 0.1693, 0.6162), (0.0320, 0.2080, 0.4652), (0.0147, 0.2586, 0.3533),
    (0.0049, 0.3230, 0.2720), (0.0024, 0.4073, 0.2123), (0.0093, 0.5030, 0.1582),
    (0.0291, 0.6082, 0.1117), (0.0633, 0.7100, 0.0782), (0.1096, 0.7932, 0.0573),
    (0.1655, 0.8620, 0.0422), (0.2257, 0.9149, 0.0298), (0.2904, 0.9540, 0.0203),
    (0.3597, 0.9803, 0.0134), (0.4334, 0.9950, 0.0087), (0.5121, 1.0000, 0.0057),
    (0.5945, 0.9950, 0.0039), (0.6784, 0.9786, 0.0027), (0.7621, 0.9520, 0.0021),
    (0.8425, 0.9154, 0.0018), (0.9163, 0.8700, 0.0017), (0.9786, 0.8163, 0.0014),
    (1.0263, 0.7570, 0.0011), (1.0567, 0.6949, 0.0010), (1.0622, 0.6310, 0.0008),
    (1.0456, 0.5668, 0.0006), (1.0026, 0.5030, 0.0003), (0.9384, 0.4412, 0.0002),
    (0.8544, 0.3810, 0.0002), (0.7514, 0.3210, 0.0001), (0.6424, 0.2650, 0.0000),
    (0.5419, 0.2170, 0.0000), (0.4479, 0.1750, 0.0000), (0.3608, 0.1382, 0.0000),
    (0.2835, 0.1070, 0.0000), (0.2187, 0.0816, 0.0000), (0.1649, 0.0610, 0.0000),
    (0.1212, 0.0446, 0.0000), (0.0874, 0.0320, 0.0000), (0.0636, 0.0232, 0.0000),
    (0.0468, 0.0170, 0.0000), (0.0329, 0.0119, 0.0000), (0.0227, 0.0082, 0.0000),
    (0.0158, 0.0057, 0.0000), (0.0114, 0.0041, 0.0000), (0.0081, 0.0029, 0.0000),
    (0.0058, 0.0021, 0.0000), (0.0041, 0.0015, 0.0000), (0.0029, 0.0010, 0.0000),
    (0.0020, 0.0007, 0.0000), (0.0014, 0.0005, 0.0000), (0.0010, 0.0004, 0.0000),
    (0.0007, 0.0002, 0.0000), (0.0005, 0.0002, 0.0000), (0.0003, 0.0001, 0.0000),
    (0.0002, 0.0001, 0.0000), (0.0002, 0.0001, 0.0000), (0.0001, 0.0000, 0.0000),
    (0.0001, 0.0000, 0.0000), (0.0001, 0.0000, 0.0000), (0.0000, 0.0000, 0.0000)))

CIE_XYZ_380_780.flags.writeable = False

WAVELENGTHS_380_780 = np.linspace(380., 780., len(CIE_XYZ_380_780))
WAVELENGTHS_380_780.flags.writeable = False


def spectrum_to_xyz(spectrum: Callable) -> ndarray:
    """
    Calculate the CIE X, Y, and Z coordinates corresponding to a light source
    with spectral distribution given by the function "spectrum", which is called
    with a series of wavelengths between 380 and 780 nm, which returns emittance
    at that wavelength in arbitrary units.  The chromaticity coordinates of the
    spectrum are returned, respecting the identity x+y+z=1.

    :param spectrum: function returning an emittance value at a given wavelength (nm)
    :return: xyz value
    """
    xyz = spectrum(WAVELENGTHS_380_780) @ CIE_XYZ_380_780
    xyz /= sum(xyz)
    return xyz


class Illuminant(Enum):
    """White point chromaticities"""
    C   = 0.3101, 0.3162             # NTSC television
    D65 = 0.3127, 0.3291             # EBU and SMPTE
    E   = 0.33333333, 0.33333333     # CIE equal-energy illuminant


# Gamma of nonlinear correction.
# See Charles Poynton's ColorFAQ Item 45 and GammaFAQ Item 6 at:
#      http://www.poynton.com/ColorFAQ.html
#      http://www.poynton.com/GammaFAQ.html
GAMMA_REC709 = 0                            # Rec. 709
GAMMA_REC709_CC = 0.018
GAMMA_REC709_FACTOR = ((1.099 * math.pow(GAMMA_REC709_CC, 0.45)) - 0.099) / GAMMA_REC709_CC


class ColorSystem:
    """
    A colour system is defined by the CIE x and y coordinates of its three
    primary illuminants and the x and y coordinates of the white point.
    """

    def __init__(self, name, red, green, blue, white, gamma):
        """
        :param name: Colour system name
        :param red: Red x, y
        :param green: Green x, y
        :param blue: Blue x, y
        :param white: White point x, y
        :param gamma: Gamma correction for system
        """
        self.name = name
        self.red = self._xyz(red)
        self.green = self._xyz(green)
        self.blue = self._xyz(blue)
        self.white = self._xyz(white.value)
        self.gamma = gamma
        self._xyz_rgb = None

    @staticmethod
    def _xyz(xy):
        return np.array((xy[0], xy[1], 1 - sum(xy)))

    @staticmethod
    def _adjugate(a):
        return la.inv(a).T * la.det(a)

    @property
    def xyz_to_rgb_matrix(self) -> ndarray:
        if self._xyz_rgb is None:

            # xyz -> rgb matrix, before scaling to white
            m = self._adjugate((self.red, self.green, self.blue))

            # White scaling factors. Dividing by yw scales the white luminance to unity
            scale = m @ self.white[:, None] / self.white[1]

            # xyz -> rgb matrix, correctly scaled to white
            m /= scale
            self._xyz_rgb = m

        return self._xyz_rgb


class ColorSystems(Enum):
    #                     name               x_red   y_red     x_green y_green   x_blue  y_blue   xy_white        gamma
    NTSC   = ColorSystem("NTSC",            (0.67,   0.33  ), (0.21,   0.71  ), (0.14,   0.08  ), Illuminant.C,   GAMMA_REC709)
    EBU    = ColorSystem("EBU (PAL/SECAM)", (0.64,   0.33  ), (0.29,   0.6   ), (0.15,   0.06  ), Illuminant.D65, GAMMA_REC709)
    SMPTE  = ColorSystem("SMPTE",           (0.63,   0.34  ), (0.31,   0.595 ), (0.155,  0.07  ), Illuminant.D65, GAMMA_REC709)
    HDTV   = ColorSystem("HDTV",            (0.67,   0.33  ), (0.21,   0.71  ), (0.15,   0.06  ), Illuminant.D65, GAMMA_REC709)
    CIE    = ColorSystem("CIE",             (0.7355, 0.2645), (0.2658, 0.7243), (0.1669, 0.0085), Illuminant.E,   GAMMA_REC709)
    REC709 = ColorSystem("CIE REC 709",     (0.64,   0.33  ), (0.3,    0.6   ), (0.15,   0.06  ), Illuminant.D65, GAMMA_REC709)

    def rgb(self, xyz: Union[ndarray, Iterable[float]]) -> ndarray:
        """
        Given an additive tricolour system CS, defined by the CIE x and y chromaticities
        of its three primaries (z is derived trivially as 1-(x+y)), and a desired
        chromaticity (XC, YC, ZC) in CIE space, determine the contribution of each
        primary in a linear combination which sums to the desired chromaticity.
        If the  requested chromaticity falls outside the Maxwell triangle (colour gamut)
        formed by the three primaries, one of the r, g, or b weights will be negative.

        Caller can use constrain_rgb() to desaturate an outside-gamut colour to the
        closest representation within the available gamut and/or norm_rgb to normalise
        the RGB components so the largest nonzero component has value 1.

        :param xyz: chromaticity in CIE space
        :return:
        """
        return self.value.xyz_to_rgb_matrix @ xyz

    def gamma_correct(self, rgb) -> ndarray:
        """
        Transform linear RGB values to nonlinear RGB values.

        Rec. 709 is ITU-R Recommendation BT. 709 (1990) ``Basic Parameter Values for
        the HDTV Standard for the Studio and for International Programme Exchange'',
        formerly CCIR Rec. 709. For details see:

            * http://www.poynton.com/ColorFAQ.html
            * http://www.poynton.com/GammaFAQ.html

        :param rgb: Linear RGB values to transform (gets modified!)
        :return: The transformed nonlinear RGB values
        """
        if self.value.gamma == GAMMA_REC709:
            # Rec.709 gamma correction
            small = rgb < GAMMA_REC709_CC
            rgb[small] *= GAMMA_REC709_FACTOR
            rgb[~small] = (1.099 * np.power(rgb[~small], 0.45)) - 0.099
        else:
            # Nonlinear colour = (Linear colour) ^ (1 / gamma)
            np.power(rgb, 1.0 / self.value.gamma, out=rgb)
        return rgb


def inside_gamut(rgb: ndarray) -> bool:
    """
    Test whether a requested colour is within the gamut achievable with the primaries
    of the current colour system. This amounts simply to testing whether all the
    primary weights are non-negative.

    :param rgb: color value to test
    :return: True when inside the gamut
    """
    return all(rgb >= 0)


class BlackBody:
    """Black body at a given temperature"""

    def __init__(self, temp: float):
        """
        :param temp: temperature of the black body in K
        """
        self.temp = temp

    def spectrum(self, wl: Union[float, ndarray]) -> Union[float, ndarray]:
        """
        Calculate, by Planck's radiation law, the emittance of a black body
        of temperature temp at the given wavelength

        :param wl: required wavelength (or wavelengths) in nm
        :return: emittance(s)
        """
        wlm = wl * 1e-9     # Wavelength to meters
        return 3.74183e-16 * wlm ** -5. / (np.exp(0.014388 / (wlm * self.temp)) - 1.)


def constrain_rgb(rgb: ndarray) -> bool:
    """
    If the requested RGB shade contains a negative weight for one of the primaries,
    it lies outside the colour gamut accessible from the given triple of primaries.
    Desaturate it by adding white, equal quantities of R, G, and B, enough to make
    RGB all positive.

    :param rgb: the color to constrain
    :return: True if the components were modified
    """
    w = - min(0, *rgb)  # Amount of white needed
    if w > 0:
        rgb += w        # Add just enough white to make r, g, b all positive
        return True     # Colour modified to fit RGB gamut
    return False        # Colour within RGB gamut


def norm_rgb(rgb: ndarray) -> ndarray:
    """
    Transform linear RGB values to nonlinear RGB values.

    Rec. 709 is ITU-R Recommendation BT. 709 (1990) ``Basic Parameter Values for
    the HDTV Standard for the Studio and for International Programme Exchange'',
    formerly CCIR Rec. 709. For details see:

       * http://www.poynton.com/ColorFAQ.html
       * http://www.poynton.com/GammaFAQ.html

    :param rgb: the color to transform (gets modified)
    :return: the modified RGB values
    """
    greatest = max(rgb)
    if greatest > 0:
        rgb /= greatest
    return rgb


if (__name__ == "__main__"):

    print("Temperature       x      y      z       R     G     B  ")
    print("-----------    ------ ------ ------   ----- ----- -----")

    for t in range(1000, 10001, 500):
        xyz = spectrum_to_xyz(BlackBody(t).spectrum)
        rgb = ColorSystems.SMPTE.rgb(xyz=xyz)
        print("  {:5.0f} K      {:.4f} {:.4f} {:.4f}   ".format(t, *xyz), end="")
        if constrain_rgb(rgb):
            fmt = "{:.3f} {:.3f} {:.3f} (Approximation)"
        else:
            fmt = "{:.3f} {:.3f} {:.3f}"
        print(fmt.format(*norm_rgb(rgb)))

    print(BlackBody(5000).spectrum(np.array([400., 500., 600.])))


    xyz = (0.3, 0.3, 0.4)
    for cs in ColorSystems:
        print(cs, cs.rgb(xyz))
    rgb = ColorSystems.CIE.rgb(xyz=(0.3, 0.3, 0.4))
    print(rgb)
    assert np.allclose([0.25455292, 0.3088893, 0.40101863], rgb)
    print(ColorSystems.CIE.value.xyz_to_rgb_matrix)

    print("gamma", ColorSystems.CIE.gamma_correct(np.array(((0.01, 0.5, 0.9), (0.018, 0.6, 1)))))

    wl = list(range(360, 820))
    s = np.array([wavelength_to_rgb(w) for w in wl]).T
    print(s.T)
    plt.plot(wl, s[0], "r", wl, s[1], "g", wl, s[2], "b")
    plt.show()


    n_palette = 12
    w0 = 400
    w1 = 800
    d = (w1 - w0) / (n_palette - 1)
    for i in range(n_palette):
        w = w0 + i * d
        print(f"{i:3.0f}, {w:3.0f}, {np.round(255*wavelength_to_rgb(w))}")
