

###########################################################
###########################################################
#       Define default Experiment names
###########################################################


_ExpWest = 'WEST'
_ExpJET = 'JET'
_ExpITER = 'ITER'
_ExpAUG = 'AUG'
_ExpDEMO = 'DEMO'
_ExpTOMAS = 'TOMAS'
_ExpCOMPASS = 'COMPASS'
_ExpCOMPASS2 = 'COMPASS2'
_ExpTCV = 'TCV'
_ExpSPARC = 'SPARC'
_ExpNSTX = 'NSTX'
_ExpKSTAR = 'KSTAR'
_ExpMAST = 'MAST'


# Default config
_DEFCONFIG = _ExpITER
# coils taken from:
# ids=['wall', 'pf_active'],
# shot=1180, run=17, database='ITER_MD', user='public'


###########################################################
###########################################################
#       Define default Configs
###########################################################


# Dictionnary of unique config names
# For each config, indicates which structural elements it comprises
# Elements are sorted by class (Ves, PFC...)
# For each element, a unique txt file containing the geometry will be loaded
_DCONFIG = {

    # WEST
    'WEST-V0': {
        'Exp': _ExpWest,
        'Ves': ['V1'],
        'CoilPF': [
            'CSV0',
            'BlV0', 'BuV0',
            'DivLow1V0', 'DivLow2V0',
            'DivUp1V0', 'DivUp2V0',
            'DlV0', 'DuV0', 'ElV0', 'EuV0', 'FlV0', 'FuV0',
        ],
    },
    'WEST-Sep': {
        'Exp': _ExpWest,
        'PlasmaDomain': ['Sep'],
        'CoilPF': [
            'CSV0',
            'BlV0', 'BuV0',
            'DivLow1V0', 'DivLow2V0',
            'DivUp1V0', 'DivUp2V0',
            'DlV0', 'DuV0', 'ElV0', 'EuV0', 'FlV0', 'FuV0',
        ],
    },
    'WEST-V1': {
        'Exp': _ExpWest,
        'Ves': ['V2'],
        'PFC': ['BaffleV0', 'DivUpV1', 'DivLowITERV1'],
        'CoilPF': [
            'CSV0',
            'BlV0', 'BuV0',
            'DivLow1V0', 'DivLow2V0',
            'DivUp1V0', 'DivUp2V0',
            'DlV0', 'DuV0', 'ElV0', 'EuV0', 'FlV0', 'FuV0',
        ],
    },
    'WEST-V2': {
        'Exp': _ExpWest,
        'Ves': ['V2'],
        'PFC': [
            'BaffleV1', 'DivUpV2', 'DivLowITERV2',
            'BumperInnerV1', 'BumperOuterV1',
            'IC1V1', 'IC2V1', 'IC3V1',
        ],
        'CoilPF': [
            'CSV0',
            'BlV0', 'BuV0',
            'DivLow1V0', 'DivLow2V0',
            'DivUp1V0', 'DivUp2V0',
            'DlV0', 'DuV0', 'ElV0', 'EuV0', 'FlV0', 'FuV0',
        ],
    },
    'WEST-V3': {
        'Exp': _ExpWest,
        'Ves': ['V2'],
        'PFC': [
            'BaffleV2', 'DivUpV3', 'DivLowITERV3',
            'BumperInnerV3', 'BumperOuterV3',
            'IC1V1', 'IC2V1', 'IC3V1',
            'LH1V1', 'LH2V1',
            'RippleV1', 'VDEV0',
        ],
        'CoilPF': [
            'CSV0',
            'BlV0', 'BuV0',
            'DivLow1V0', 'DivLow2V0',
            'DivUp1V0', 'DivUp2V0',
            'DlV0', 'DuV0', 'ElV0', 'EuV0', 'FlV0', 'FuV0',
        ],
    },
    'WEST-V4': {
        'Exp': _ExpWest,
        'Ves': ['InnerV0', 'OuterV0'],
        'PFC': [
            'BaffleV2', 'DivUpV3', 'DivLowITERV3',
            'BumperInnerV3', 'BumperOuterV3',
            'RippleV1', 'VDEV0',
            'ThermalShieldHFSV0',
            'ThermalShieldLFSSlimV0',
            'ThermalShieldLFSWideV0',
            'ThermalShieldLFSLowV0',
            'ThermalShieldLFSUpV0',
            'LH1V1', 'LH2V1',
            'IC1V1', 'IC2V1', 'IC3V1',
            'CasingCoverLDivV0', 'CasingCoverUDivV0',
            'CasingLDivV0', 'CasingUDivV0',
            'CasingPFUPlateLDivV0', 'CasingPFUPlateUDivV0',
            'CasingPJLDivV0', 'CasingPJUDivV0',
        ],
        'CoilPF': [
            'CSV0',
            'BlV0', 'BuV0',
            'DivLow1V0', 'DivLow2V0',
            'DivUp1V0', 'DivUp2V0',
            'DlV0', 'DuV0', 'ElV0', 'EuV0', 'FlV0', 'FuV0',
        ],
    },

    # JET
    'JET-V0': {
        'Exp': _ExpJET,
        'Ves': ['V0'],
    },

    # ITER
    'ITER-V0': {
        'Exp': _ExpITER,
        'Ves': ['V0'],
        'CoilPF': ['PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6'],
        'CoilCS': ['CS3U', 'CS2U', 'CS1U', 'CS1L', 'CS2L', 'CS3L'],
    },
    'ITER-V1': {
        'Exp': _ExpITER,
        'Ves': ['V1'],
        'PFC': [
            'BLK01', 'BLK02', 'BLK03', 'BLK04', 'BLK05', 'BLK06',
            'BLK07', 'BLK08', 'BLK09', 'BLK10', 'BLK11', 'BLK12',
            'BLK13', 'BLK14', 'BLK15', 'BLK16', 'BLK17', 'BLK18',
            'Div1', 'Div2', 'Div3', 'Div4', 'Div5', 'Div6',
        ],
        'CoilPF': ['PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6'],
        'CoilCS': ['CS3U', 'CS2U', 'CS1U', 'CS1L', 'CS2L', 'CS3L'],
    },
    'ITER-V2': {
        'Exp': _ExpITER,
        'Ves': ['InnerV0', 'OuterV0', 'Cryostat'],
        'PFC': [
            'BLK01', 'BLK02', 'BLK03', 'BLK04', 'BLK05', 'BLK06',
            'BLK07', 'BLK08', 'BLK09', 'BLK10', 'BLK11', 'BLK12',
            'BLK13', 'BLK14', 'BLK15', 'BLK16', 'BLK17', 'BLK18',
            'Div1', 'Div2', 'Div3', 'Div4', 'Div5', 'Div6',
        ],
        'CoilPF': ['PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6'],
        'CoilCS': ['CS3U', 'CS2U', 'CS1U', 'CS1L', 'CS2L', 'CS3L'],
    },
    # 'ITER-SOLEDGE3XV0': {
    # 'Exp': _ExpITER,
    # 'Ves': ['SOLEDGE3XV0'],
    # 'PFC': ['SOLEDGE3XDivDomeV0', 'SOLEDGE3XDivSupportV0']
    # },

    # AUG
    'AUG-V0': {
        'Exp': _ExpAUG,
        'Ves': ['V0'],
    },
    'AUG-V1': {
        'Exp': _ExpAUG,
        'Ves': ['VESiR'],
        'PFC': [
            'D2cdome', 'D2cdomL', 'D2cdomR', 'D2ci1',
            'D2ci2', 'D2cTPib', 'D2cTPic', 'D2cTPi',
            'D2dBG2', 'D2dBl1', 'D2dBl2', 'D2dBl3',
            'D2dBu1', 'D2dBu2', 'D2dBu3', 'D2dBu4',
            'D3BG10', 'D3BG1', 'ICRHa', 'LIM09', 'PClow',
            'PCup', 'SBi', 'TPLT1', 'TPLT2', 'TPLT3',
            'TPLT4', 'TPLT5', 'TPRT2', 'TPRT3', 'TPRT4',
            'TPRT5',
        ],
    },

    # DEMO
    'DEMO-2019': {
        'Exp': _ExpDEMO,
        'Ves': ['V0'],
        'PFC': [
            'LimiterUpperV0', 'LimiterEquatV0',
            'BlanketInnerV0', 'BlanketOuterV0', 'DivertorV0'],
    },

    # TOMAS
    'TOMAS-V0': {
        'Exp': _ExpTOMAS,
        'Ves': ['V0'],
        'PFC': ['LimiterV0', 'AntennaV0'],
    },

    # COMPASS
    'COMPASS-V0': {
        'Exp': _ExpCOMPASS,
        'Ves': ['V0'],
    },
    'COMPASS-V1': {
        'Exp': _ExpCOMPASS,
        'Ves': ['InnerV1'],
        'PFC': ['lower', 'upper', 'inner', 'outer'],
    },

    # COMPASS2
    'COMPASS2-V0': {
        'Exp': _ExpCOMPASS2,
        'Ves': ['V0'],
    },

    # TCV
    'TCV-V0': {
        'Exp': _ExpTCV,
        'Ves': ['vIn', 'vOut', 't'],
        'CoilPF': [
            'A001', 'B001', 'B002',
            'B03A1', 'B03A2', 'B03A3',
            'C001', 'C002',
            'D001', 'D002',
            'E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E008',
            'E03A1', 'E03A2', 'E03A3',
            'F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008',
            'T03A1', 'T03A2', 'T03A3'],
    },

    # SPARC
    'SPARC-V0': {
        'Exp': _ExpSPARC,
        'Ves': ['FirstWallV0'],
    },

    'SPARC-V1': {
        'Exp': _ExpSPARC,
        'Ves': ['FirstWallV0', 'VesInner', 'VesOuter'],
        'PFC': ['ICRH0'],
        'CoilPF': [
            'Div1lower', 'Div2lower', 'Div1upper', 'Div2upper',
            'EFClower0', 'EFClower1', 'EFCmed0', 'EFCmed1',
            'EFCupper0', 'EFCupper1', 'PF1lower', 'PF1upper',
            'PF2lower', 'PF2upper', 'PF3lower', 'PF3upper',
            'PF4lower', 'PF4upper', 'VS1lower', 'VS1upper',
            'VStabPlatelower', 'VStabPlateupper',
        ],
        'CoilCS': [
            'CS1lower', 'CS2lower', 'CS3lower',
            'CS1upper', 'CS2upper', 'CS3upper',
        ],
    },

    'SPARC-V2': {
        'Exp': _ExpSPARC,
        'Ves': [
            'FirstWallV0', 'VesInner', 'VesOuter',
            'CoilTFInner', 'CoilTFOuter',
        ],
        'PFC': ['ICRH0'],
        'CoilPF': [
            'Div1lower', 'Div2lower', 'Div1upper', 'Div2upper',
            'EFClower0', 'EFClower1', 'EFCmed0', 'EFCmed1',
            'EFCupper0', 'EFCupper1', 'PF1lower', 'PF1upper',
            'PF2lower', 'PF2upper', 'PF3lower', 'PF3upper',
            'PF4lower', 'PF4upper', 'VS1lower', 'VS1upper',
            'VStabPlatelower', 'VStabPlateupper',
        ],
        'CoilCS': [
            'CS1lower', 'CS2lower', 'CS3lower',
            'CS1upper', 'CS2upper', 'CS3upper',
        ],
    },

    # NSTX
    'NSTX-V0': {
        'Exp': _ExpNSTX,
        'Ves': ['V0'],
    },
    'NSTX-V1': {
        'Exp': _ExpNSTX,
        'Ves': ['VesselInner'],
        'CoilPF': [
            'CentralSolenoid', 'PFCoil01', 'PFCoil02', 'PFCoil03',
            'PFCoil04', 'PFCoil05', 'PFCoil06', 'PFCoil07', 'PFCoil08',
            'PFCoil09', 'PFCoil10', 'PFCoil11', 'PFCoil12', 'PFCoil13',
            'PFCoil14', 'PFCoil16', 'PFCoil17', 'PFCoil18', 'PFCoil19',
            'PFCoil20', 'PFCoil21', 'PFCoil22',
        ],
    },
    'NSTX-V2': {
        'Exp': _ExpNSTX,
        'Ves': ['VesselInner'],
        'PFC': [
            'VesselOutter01', 'VesselOutter02',
            'BumperOutter01', 'BumperOutter02',
            'BumperOutter03', 'BumperOutter04',
            'DivertorUpper', 'DivertorLower',
            'ICRFAntenna',
            'path69778', 'path69912', 'path69980', 'path70015',
            'path70085', 'path70153',
        ],
        'CoilPF': [
            'CentralSolenoid', 'PFCoil01', 'PFCoil02', 'PFCoil03',
            'PFCoil04', 'PFCoil05', 'PFCoil06', 'PFCoil07', 'PFCoil08',
            'PFCoil09', 'PFCoil10', 'PFCoil11', 'PFCoil12', 'PFCoil13',
            'PFCoil14', 'PFCoil16', 'PFCoil17', 'PFCoil18', 'PFCoil19',
            'PFCoil20', 'PFCoil21', 'PFCoil22',
        ],
    },

    # KSTAR
    'KSTAR-V0': {
        'Exp': _ExpKSTAR,
        'Ves': ['V0'],
    },

    # MAST
    'MAST-V0': {
        'Exp': _ExpMAST,
        'Ves': ['V0'],
    },
}

# Each config can be called by various names / shortcuts (for benchmark and
# retro-compatibility), this table stores, for each shortcut,
# the associated unique name it refers to
_DCONFIG_SHORTCUTS = {
    'ITER': 'ITER-V2',
    # 'ITER-SOLEDGE3X': 'ITER-SOLEDGE3XV0',
    'JET': 'JET-V0',
    'WEST': 'WEST-V4',
    'A1': 'WEST-V1',
    'A2': 'ITER-V1',
    'A3': 'WEST-Sep',
    'B1': 'WEST-V2',
    'B2': 'WEST-V3',
    'B3': 'WEST-V4',
    'B4': 'ITER-V2',
    'AUG': 'AUG-V1',
    'DEMO': 'DEMO-2019',
    'TOMAS': 'TOMAS-V0',
    'COMPASS': 'COMPASS-V0',
    'TCV': 'TCV-V0',
    'SPARC': 'SPARC-V1',
    'NSTX': 'NSTX-V2',
    'KSTAR': 'KSTAR-V0',
    'MAST': 'MAST-V0',
}


# Check all shortcuts
lout = []
for k0, v0 in _DCONFIG_SHORTCUTS.items():
    if v0 not in _DCONFIG.keys():
        lout.append('{}: {}'.format(k0, v0))
if len(lout) > 0:
    msg = (
        "\nThe following shortcuts refer to undefined config names:\n"
        + "\t- "
        + "\n\t- ".join(lout)
        + "\nAvailable config names are:\n"
        + "\t- "
        + "\n\t- ".join(_DCONFIG.keys())
    )
    raise Exception(msg)
