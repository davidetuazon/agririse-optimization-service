"""
TODO:
    1. get radiation and windspeed readings
"""

def compute_et0(temperature_c: float, humidity_percent: float, radiation_mj_m2: float = 15, wind_speed_m_s: float = 2) -> float:
    """
    Using simplified FAO Penman-Monteith Method
    """
    # percentage to fraction
    rh = humidity_percent / 100

    # saturation vapor pressure using temperature
    #     const     const       const                                     const
    es = 0.6108 * (2.71828 ** ((17.27 * temperature_c) / (temperature_c + 237.3)))

    # actual vapor pressure
    ea = rh * es

    gamma = 0.066 # kPa/degC, approximate
    #       const                         const    const
    delta = 4098 * es / ((temperature_c + 273.3) ** 2)

    Rn_minus_G = radiation_mj_m2 # placeholder
    
    #           const
    wind_term = 0.26 * wind_speed_m_s * (es - ea)

    #      const                                                                const
    et0 = (0.408 * delta * Rn_minus_G + gamma * wind_term) / (delta + gamma * (1 + 0.34 * wind_speed_m_s))

    return max(et0, 0)