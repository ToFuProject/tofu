
import os
import sys
import argparse
import datetime as dtm


import numpy as np


_PATH_TOFU = os.path.join(
    os.path.expanduser('~'),
    'projects',
    'tofu',
)
sys.path.insert(0, _PATH_TOFU)
import tofu as tf


# ###########################################
# ###########################################
#              DEFAULTS
# ###########################################


_DDEF = {
    'd2cross': '',
    'nEph': 241,
    'nEe0': 240,
    'ntheta_ph_vsB': 91,
    'ntheta_e0_vsB': 93,
    'nphi_e0_vsB': 181,
}


# ###########################################
# ###########################################
#              Main
# ###########################################


def main(
    d2cross=None,
    nEph=None,
    nEe0=None,
    ntheta_ph_vsB=None,
    ntheta_e0_vsB=None,
    nphi_e0_vsB=None,
    ddef=None,
):

    # ------------------
    # timing

    t0 = dtm.datetime.now()

    # -------------------
    # msg

    lstr = [f"\t- {k0}: {v0}" for k0, v0 in locals().items()]
    msg = (
        f"\ntofu file: {tf.__file__}\n\n"
        f"Script: {__file__}\n"
        "Input args:\n"
        + "\n".join(lstr)
    )
    print(msg)

    # -------------------
    # inputs

    if d2cross is None:
        d2cross = ddef['d2cross']

    if nEph is None:
        nEph = ddef['nEph']
    if nEe0 is None:
        nEe0 = ddef['nEe0']
    if ntheta_ph_vsB is None:
        ntheta_ph_vsB = ddef['ntheta_ph_vsB']
    if ntheta_e0_vsB is None:
        ntheta_e0_vsB = ddef['ntheta_e0_vsB']
    if nphi_e0_vsB is None:
        nphi_e0_vsB = ddef['nphi_e0_vsB']

    E_ph_eV = np.logspace(0, 8, nEph)
    E_e0_eV = np.logspace(0, 8, nEe0)
    theta_ph_vsB = np.linspace(0, np.pi, ntheta_ph_vsB)

    # -------------------
    # call

    _mod = tf.physics_tools.electrons.emission
    d2cross_phi = _mod.get_d2cross_phi(
        d2cross=d2cross,
        E_ph_eV=E_ph_eV,
        E_e0_eV=E_e0_eV,
        theta_ph_vsB=theta_ph_vsB,
        theta_e0_vsB_npts=ntheta_e0_vsB,
        phi_e0_vsB_npts=nphi_e0_vsB,
        save=True,
        verb=2,
    )

    # ------------------
    # timing

    dt = (dtm.datetime.now() - t0).total_seconds()
    msg = f"\nCPU time = {dt/60} min"
    print(msg)

    return


# ###########################################
# ###########################################
#           __main__
# ###########################################


if __name__ == '__main__':

    # -----------
    # default
    # -----------

    msg = (
        "Tabulate d2cross over desired grid (E_ph, E_e0, theta_ph)"
    )

    # -----------
    # parse args
    # -----------

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # d2cross
    parser.add_argument(
        '-d2cross',
        '--d2cross',
        type=str,
        help='<path/file.ext> to an existing d2cross_...npz tabulation',
        required=False,
        default=_DDEF['d2cross'],
    )

    # nEph
    parser.add_argument(
        '-nEph',
        '--nEph',
        type=int,
        help='Number of np.logspace(0, 8, nEph) (eV)',
        required=False,
        default=_DDEF['nEph'],
    )

    # nEe0
    parser.add_argument(
        '-nEe0',
        '--nEe0',
        type=int,
        help='Number of np.logspace(0, 8, nEe0) (eV)',
        required=False,
        default=_DDEF['nEe0'],
    )

    # ntheta_ph_vsB
    parser.add_argument(
        '-ntheta_ph_vsB',
        '--ntheta_ph_vsB',
        type=int,
        help='Number of np.linspace(0, np.pi, ntheta) (rad)',
        required=False,
        default=_DDEF['ntheta_ph_vsB'],
    )

    # ntheta_e0_vsB
    parser.add_argument(
        '-ntheta_e0_vsB',
        '--ntheta_e0_vsB',
        type=int,
        help='Number of np.linspace(0, np.pi, ntheta) (rad)',
        required=False,
        default=_DDEF['ntheta_e0_vsB'],
    )

    # nphi_e0_vsB
    parser.add_argument(
        '-nphi_e0_vsB',
        '--nphi_e0_vsB',
        type=int,
        help='Number of np.linspace(-np.pi, np.pi, ntheta) (rad)',
        required=False,
        default=_DDEF['nphi_e0_vsB'],
    )

    # -----------
    # call main
    # -----------

    # Parse arguments
    args = parser.parse_args()

    # Call function
    main(ddef=_DDEF, **dict(args._get_kwargs()))
