from astropy.table import Table, dstack
import numpy as np


def read_ds_data_T(data_path_template, m_bins, **kwargs):
    table_list = []
    for i in np.arange(len(m_bins) - 1):
        table_tmp = Table.read(data_path_template % i, **kwargs)
        table_tmp["mass_bin"] = i
        table_list.append(table_tmp)
    out = dstack(table_list)

    # convert from Mpc**2 to pc**2 units
    out["ds_t_pc2"] = out["ds_t"] * 10**12
    if "ds_err" in out.colnames:
        out["ds_err_pc2"] = out["ds_err"] * 10**12

    return out


def read_ds_data(data_path_template, m_bins, **kwargs):
    table_dict = {}
    for i in np.arange(len(m_bins) - 1):
        print(data_path_template % i)
        table_tmp = Table.read(data_path_template % i, **kwargs)
        table_tmp["mass_bin"] = i
        for col in table_tmp.colnames:
            table_dict[col] = table_dict.get(col, []) + [table_tmp[col]]
    # out = Table(list(table_dict.values()), names=list(table_dict.keys()))
    out = Table(table_dict)

    # convert from Mpc**2 to pc**2 units
    out["ds_t_pc2"] = out["ds_t"] * 10**12
    if "ds_err" in out.colnames:
        out["ds_err_pc2"] = out["ds_err"] * 10**12

    return out


def nz_normal(*args):
    val = np.random.normal(*args)
    while val < 0:
        val = np.random.normal(*args)
    return val
