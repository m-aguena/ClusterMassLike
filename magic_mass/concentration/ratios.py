import numpy as np
from scipy.optimize import fsolve, least_squares
from ..nfw import nfw_int


def get_rho0_rs_v0(M1, R1, M2, R2, Rs0=20):
    def func(Rs):
        # if Rs<=0:
        #    return np.nan
        return abs(nfw_int(R1 / Rs) / nfw_int(R2 / Rs) - M1 / M2)

    Rs = fsolve(func, x0=Rs0)

    rho0 = M1 / (4 * np.pi * Rs**3) / nfw_int(R1 / Rs)

    # rho0_2 = M2/(4 * np.pi * Rs**3)/nfw_int(R2/Rs)
    # print(rho0_2/rho0-1)

    print("c1, c2 :", R1 / Rs, R2 / Rs)
    print("diff   :", nfw_int(R1 / Rs) / nfw_int(R2 / Rs) - M1 / M2)

    return Rs, rho0


def get_rho0_rs(M1, R1, M2, R2, Rs0=20):
    # print(M1, R1, M2, R2)

    res = least_squares(
        lambda Rs: np.sqrt((nfw_int(R1 / Rs) / nfw_int(R2 / Rs) - M1 / M2) ** 2),
        x0=Rs0,
        bounds=(0.001, 20),
    )
    # print(res)
    Rs = res.x

    rho0 = M1 / (4 * np.pi * Rs**3) / nfw_int(R1 / Rs)

    # rho0_2 = M2/(4 * np.pi * Rs**3)/nfw_int(R2/Rs)
    # print(rho0_2/rho0-1)

    # print("c1, c2 :", R1 / Rs, R2 / Rs)
    # print("diff   :", nfw_int(R1 / Rs) / nfw_int(R2 / Rs) - M1 / M2)

    return Rs, rho0


def get_rho0_rs_safe(M1, R1, M2, R2, Rs0=20):
    mask = ~(np.isnan(M1) + np.isnan(R1) + np.isnan(M2) + np.isnan(R2))

    Rs = np.full(mask.size, np.nan)
    rho0 = np.full(mask.size, np.nan)

    Rs[mask], rho0[mask] = get_rho0_rs(
        M1[mask], R1[mask], M2[mask], R2[mask], np.array(Rs0)[mask]
    )

    return Rs, rho0


def add_conc_rs_rho0_to_bs_table(bs_table, Delta1, Delta2):
    bs_table["Rs"], bs_table["rho0"] = np.array(
        [
            get_rho0_rs_safe(m200, r200, m500, r500, Rs0=20 * np.ones(m200.size))
            for m200, r200, m500, r500 in zip(
                np.array(bs_table[f"M{Delta1}_crit"]).T,
                np.array(bs_table[f"R{Delta1}_crit"]).T,
                np.array(bs_table[f"M{Delta2}_crit"]).T,
                np.array(bs_table[f"R{Delta2}_crit"]).T,
            )
        ]
    ).transpose(1, 2, 0)

    # bs_table[f"c{Delta2}_crit"] = bs_table[f"R{Delta2}_crit"] / bs_table["Rs"]
    # bs_table[f"c{Delta1}_crit"] = bs_table[f"R{Delta1}_crit"] / bs_table["Rs"]

    bs_table[f"c{Delta2}_crit"] = bs_table[f"R{Delta2}_crit"] / np.nanmean(
        bs_table["Rs"], axis=0
    )
    bs_table[f"c{Delta1}_crit"] = bs_table[f"R{Delta1}_crit"] / np.nanmean(
        bs_table["Rs"], axis=0
    )

    bs_table["Rs_chi2"] = (
        nfw_int(bs_table[f"R{Delta1}_crit"] / bs_table["Rs"])
        / nfw_int(bs_table[f"R{Delta2}_crit"] / bs_table["Rs"])
        - bs_table[f"M{Delta1}_crit"] / bs_table[f"M{Delta2}_crit"]
    ) ** 2


def add_conc_rs_rho0_to_full_table(full_table, Delta1, Delta2):
    full_table["Rs"], full_table["rho0"] = get_rho0_rs(
        np.array(full_table[f"M{Delta1}_crit"]),
        np.array(full_table[f"R{Delta1}_crit"]),
        np.array(full_table[f"M{Delta2}_crit"]),
        np.array(full_table[f"R{Delta2}_crit"]),
        Rs0=20 * np.ones(len(full_table)),
    )

    full_table[f"c{Delta2}_crit"] = full_table[f"R{Delta2}_crit"] / full_table["Rs"]
    full_table[f"c{Delta1}_crit"] = full_table[f"R{Delta1}_crit"] / full_table["Rs"]

    full_table["Rs_chi2"] = (
        nfw_int(full_table[f"R{Delta1}_crit"] / full_table["Rs"])
        / nfw_int(full_table[f"R{Delta2}_crit"] / full_table["Rs"])
        - full_table[f"M{Delta1}_crit"] / full_table[f"M{Delta2}_crit"]
    ) ** 2


# #### Errs for full_table


def get_rs_err(M1, R1, M2, R2, Rs, M1_err, M2_err):
    return (
        Rs
        * nfw_int(R1 / Rs)
        * np.sqrt(
            ((M1_err / M1) ** 2 + (M2_err / M2) ** 2)
            / ((R1 / (Rs + R1)) ** 4 + (M1 / M2) ** 2 * (R2 / (Rs + R2)) ** 4)
        )
    )


def add_conc_rs_err_to_full_table(full_table, Delta1, Delta2):
    full_table["Rs_err"] = get_rs_err(
        *(
            np.array(v)
            for v in (
                full_table[f"M{Delta1}_crit"],
                full_table[f"R{Delta1}_crit"],
                full_table[f"M{Delta2}_crit"],
                full_table[f"R{Delta2}_crit"],
                full_table["Rs"],
                full_table[f"M{Delta1}_crit_err"],
                full_table[f"M{Delta2}_crit_err"],
            )
        )
    )

    full_table[f"c{Delta2}_crit_err"] = (
        full_table[f"c{Delta2}_crit"] * full_table["Rs_err"] / full_table["Rs"]
    )
    full_table[f"c{Delta1}_crit_err"] = (
        full_table[f"c{Delta1}_crit"] * full_table["Rs_err"] / full_table["Rs"]
    )
