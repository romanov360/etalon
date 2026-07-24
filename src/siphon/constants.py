"""Physical constants used throughout SiPhon (SI units unless noted)."""

C_UM_PER_S = 2.99792458e14  # speed of light, um/s
C_M_PER_S = 2.99792458e8  # speed of light, m/s
H_PLANCK_J_S = 6.62607015e-34  # Planck constant, J*s
Q_ELECTRON_C = 1.602176634e-19  # elementary charge, C
K_BOLTZMANN_J_PER_K = 1.380649e-23  # Boltzmann constant, J/K

# Common telecom bands (wavelength in um)
O_BAND_UM = (1.260, 1.360)
C_BAND_UM = (1.530, 1.565)
L_BAND_UM = (1.565, 1.625)


def photon_energy_j(wavelength_um: float) -> float:
    """Photon energy in joules at a given vacuum wavelength in um."""
    return H_PLANCK_J_S * C_M_PER_S / (wavelength_um * 1e-6)


def db_to_linear(db: float) -> float:
    """Convert a power ratio in dB to linear scale."""
    return 10.0 ** (db / 10.0)


def linear_to_db(linear) -> float:
    """Convert a linear power ratio to dB."""
    import numpy as np

    return 10.0 * np.log10(linear)


def dbm_to_mw(dbm: float) -> float:
    """Convert power in dBm to mW."""
    return 10.0 ** (dbm / 10.0)


def mw_to_dbm(mw) -> float:
    """Convert power in mW to dBm."""
    import numpy as np

    return 10.0 * np.log10(mw)
