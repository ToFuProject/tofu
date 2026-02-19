
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
#              Main
# ###########################################


def main(
    nEph=None,
    nEe0=None,
    ntheta=None,
    version=None,
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

    if nEph is None:
        nEph = ddef['nEph']
    if nEe0 is None:
        nEe0 = ddef['nEe0']
    if ntheta is None:
        ntheta = ddef['ntheta']
    if version is None:
        version = ddef['version']

    E_ph_eV = np.logspace(0, 8, nEph)
    E_e0_eV = np.logspace(0, 8, nEe0)
    theta_ph = np.linspace(0, np.pi, ntheta)

    # -------------------
    # call

    _mod = tf.physics_tools.electrons.emission
    d2cross = _mod.get_xray_thin_d2cross_ei_integrated_thetae_dphi(
        E_e0_eV=E_e0_eV[None, :, None],
        E_ph_eV=E_ph_eV[None, None, :],
        theta_ph=theta_ph[:, None, None],
        save=True,
        verb=2,
        version='BHE',
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

    ddef = {
        'nEph': 401,
        'nEe0': 400,
        'ntheta': 181,
        'version': 'EH',
    }

    # -----------
    # parse args
    # -----------

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # nEph
    parser.add_argument(
        '-nEph',
        '--nEph',
        type=int,
        help='Number of np.logspace(0, 8, nEph) (eV)',
        required=False,
        default=ddef['nEph'],
    )

    # nEe0
    parser.add_argument(
        '-nEe0',
        '--nEe0',
        type=int,
        help='Number of np.logspace(0, 8, nEe0) (eV)',
        required=False,
        default=ddef['nEe0'],
    )

    # ntheta
    parser.add_argument(
        '-ntheta',
        '--ntheta',
        type=int,
        help='Number of np.linspace(0, np.pi, ntheta) (rad)',
        required=False,
        default=ddef['ntheta'],
    )

    # version
    parser.add_argument(
        '-v',
        '--version',
        type=str,
        help="version of the cross-section in ['BHE', 'EH']",
        required=False,
        default=ddef['version'],
    )

    # -----------
    # call main
    # -----------

    # Parse arguments
    args = parser.parse_args()

    # Call function
    main(ddef=ddef, **dict(args._get_kwargs()))
