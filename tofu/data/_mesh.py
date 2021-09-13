# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np


# tofu
# from tofu import __version__ as __version__
import tofu.utils as utils
from . import _core_new






# #############################################################################
# #############################################################################
#                           Mesh2DRect
# #############################################################################


class Mesh2DRect(_core_new.DataCollection):

    _ddef = {
        'Id': {'include': ['Mod', 'Cls', 'Name', 'version']},
        'params': {
            'lambda0': (float, 0.),
            'source': (str, 'unknown'),
            'transition':    (str, 'unknown'),
            'element':  (str, 'unknown'),
            'charge':  (int, 0),
            'ion':  (str, 'unknown'),
            'symbol':   (str, 'unknown'),
        },
    }
    _forced_group = [_GROUP_NE, _GROUP_TE]
    _data_none = True

    _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'

    _grouplines = _GROUP_LINES
    _groupne = _GROUP_NE
    _groupte = _GROUP_TE

    _units_lambda0 = _UNITS_LAMBDA0


    def add_mesh(
        self,
        key=None,
        lambda0=None,
        pec=None,
        source=None,
        transition=None,
        ion=None,
        symbol=None,
        **kwdargs,
    ):
        """ Add a mesh by key

        """
        self.add_obj(
            which='lines',
            key=key,
            lambda0=lambda0,
            pec=pec,
            source=source,
            transition=transition,
            ion=ion,
            symbol=symbol,
            **kwdargs,
        )
        pass



    # -----------------
    # from config
    # ------------------

    @classmethod
    def _from_Config(
        cls,
        config=None,
        dsource0=None,
        dref0=None,
        ddata0=None,
        dlines0=None,
        grouplines=None,
    ):
        """

        Example:
        --------
                >>> import tofu as tf
                >>> conf = tf.load_config('ITER')
                >>> mesh = tf.data.Mesh2DRect.from_Config(
                    config=conf,
                    res=[],
                )

        """

        # Preliminary import and checks

        # Load from online if relevant

        # Load for local files
        dne, dte, dpec, lion, dsource, dlines = _read_files.step03_read_all(
            lambmin=lambmin,
            lambmax=lambmax,
            element=element,
            charge=charge,
            pec_as_func=False,
            format_for_DataCollection=True,
            dsource0=dsource0,
            dref0=dref0,
            ddata0=ddata0,
            dlines0=dlines0,
            verb=False,
        )

        # # dgroup
        # dgroup = ['Te', 'ne']

        # dref - Te + ne
        dref = dte
        dref.update(dne)

        # ddata - pec
        ddata = dpec

        # dref_static
        dref_static = {
            'ion': {k0: {} for k0 in lion},
            'source': dsource,
        }

        # dobj (lines)
        dobj = {
            grouplines: dlines,
        }
        return ddata, dref, dref_static, dobj

    @classmethod
    def from_Config(
        cls,
        config=None,
    ):
        """
        Load lines and pec from openadas, either:
            - online = True:  directly from the website
            - online = False: from pre-downloaded files in ~/.tofu/openadas/
        """
        ddata, dref, dref_static, dobj = cls._from_openadas(
            lambmin=lambmin,
            lambmax=lambmax,
            element=element,
            charge=charge,
            online=online,
            update=update,
            create_custom=create_custom,
            grouplines=grouplines,
        )
        return cls(ddata=ddata, dref=dref, dref_static=dref_static, dobj=dobj)
