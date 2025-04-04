""" A tomography library for fusion devices (tokamaks)

See:
https://github.com/ToFuProject/tofu
"""

# Built-in
import os
import glob
import shutil
import logging
import platform
import subprocess
from codecs import open
# ... setup tools
from setuptools import setup, find_packages
# ... for `clean` command
from distutils.command.clean import clean as Clean


# ... packages that need to be in pyproject.toml
import numpy as np
from Cython.Distutils import Extension
from Cython.Distutils import build_ext


# ... local script
import _updateversion as up
# ... openmp utilities
from tofu_helpers.openmp_helpers import is_openmp_installed


# == Checking platform ========================================================
is_platform_windows = platform.system() == "Windows"


# === Setting clean command ===================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tofu.setup")


class CleanCommand(Clean):

    description = "Remove build artifacts from the source tree"

    def expand(self, path_list):
        """
        Expand a list of path using glob magic.
        :param list[str] path_list: A list of path which may contains magic
        :rtype: list[str]
        :returns: A list of path without magic
        """
        path_list2 = []
        for path in path_list:
            if glob.has_magic(path):
                iterator = glob.iglob(path)
                path_list2.extend(iterator)
            else:
                path_list2.append(path)
        return path_list2

    def find(self, path_list):
        """Find a file pattern if directories.
        Could be done using "**/*.c" but it is only supported in Python 3.5.
        :param list[str] path_list: A list of path which may contains magic
        :rtype: list[str]
        :returns: A list of path without magic
        """
        import fnmatch

        path_list2 = []
        for pattern in path_list:
            for root, _, filenames in os.walk("."):
                for filename in fnmatch.filter(filenames, pattern):
                    path_list2.append(os.path.join(root, filename))
        return path_list2

    def run(self):
        Clean.run(self)

        cython_files = self.find(["*.pyx"])
        cythonized_files = [
            path.replace(".pyx", ".c") for path in cython_files
        ]
        so_files = self.find(["*.so"])
        # really remove the directories
        # and not only if they are empty
        to_remove = [self.build_base]
        to_remove = self.expand(to_remove)
        to_remove += cythonized_files
        to_remove += so_files

        if not self.dry_run:
            for path in to_remove:
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    logger.info("removing '%s'", path)
                except OSError:
                    pass
# =============================================================================


# == Getting tofu version =====================================================
_HERE = os.path.abspath(os.path.dirname(__file__))


def get_version_tofu(path=_HERE):

    # Try from git
    isgit = ".git" in os.listdir(path)
    if isgit:
        try:
            git_branch = (
                subprocess.check_output(
                    [
                        "git",
                        "rev-parse",
                        "--abbrev-ref",
                        "HEAD",
                    ]
                )
                .rstrip()
                .decode()
            )
            deploy_branches = ["master", "deploy-test"]
            if (git_branch in deploy_branches or "TRAVIS_TAG" in os.environ):
                version_tofu = up.updateversion()
            else:
                isgit = False
        except Exception:
            isgit = False

    if not isgit:
        version_tofu = os.path.join(path, "tofu")
        version_tofu = os.path.join(version_tofu, "version.py")
        with open(version_tofu, "r") as fh:
            version_tofu = fh.read().strip().split("=")[-1].replace("'", "")

    version_tofu = version_tofu.lower().replace("v", "").replace(" ", "")
    return version_tofu


version_tofu = get_version_tofu(path=_HERE)

print("")
print("Version for setup.py : ", version_tofu)
print("")

# =============================================================================

# =============================================================================
# Get the long description from the README file
# Get the readme file whatever its extension (md vs rst)

_README = [
    ff
    for ff in os.listdir(_HERE)
    if len(ff) <= 10 and ff[:7] == "README."
]
assert len(_README) == 1
_README = _README[0]
with open(os.path.join(_HERE, _README), encoding="utf-8") as f:
    long_description = f.read()
if _README[-3:] == ".md":
    long_description_content_type = "text/markdown"
else:
    long_description_content_type = "text/x-rst"
# =============================================================================


# =============================================================================
#  Compiling files
openmp_installed, openmp_flag = is_openmp_installed()

extra_compile_args = ["-O3", "-Wall", "-fno-wrapv"] + openmp_flag
extra_link_args = [] + openmp_flag

extensions = [
    Extension(
        name="tofu.geom._GG",
        sources=["tofu/geom/_GG.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language_level="3",
    ),
    Extension(
        name="tofu.geom._basic_geom_tools",
        sources=["tofu/geom/_basic_geom_tools.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._distance_tools",
        sources=["tofu/geom/_distance_tools.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._sampling_tools",
        sources=["tofu/geom/_sampling_tools.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._raytracing_tools",
        sources=["tofu/geom/_raytracing_tools.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._vignetting_tools",
        sources=["tofu/geom/_vignetting_tools.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._chained_list",
        sources=["tofu/geom/_chained_list.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._sorted_set",
        sources=["tofu/geom/_sorted_set.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="tofu.geom._openmp_tools",
        sources=["tofu/geom/_openmp_tools.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cython_compile_time_env=dict(TOFU_OPENMP_ENABLED=openmp_installed),
    ),
]


setup(
    name="tofu",
    version="{ver}".format(ver=version_tofu),
    # Use scm to get code version from git tags
    # cf. https://pypi.python.org/pypi/setuptools_scm
    # Versions should comply with PEP440. For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    # The version is stored only in the setup.py file and read from it (option
    # 1 in https://packaging.python.org/en/latest/single_source_version.html)
    use_scm_version=False,

    # Description of what tofu does
    description="A python library for Tomography for Fusion",
    long_description=long_description,
    long_description_content_type=long_description_content_type,

    # The project's main homepage.
    url="https://github.com/ToFuProject/tofu",
    # Author details
    author="Didier VEZINET and Laura MENDOZA",
    author_email="didier.vezinet@gmail.com",

    # Choose your license
    license="MIT",

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        # In which language most of the code is written ?
        "Natural Language :: English",
    ],

    # What does your project relate to?
    keywords="tomography geometry 3D inversion synthetic fusion",

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(
        exclude=[
            "doc",
            "_Old",
            "_Old_doc",
            "plugins",
            "plugins.*",
            "*.plugins.*",
            "*.plugins",
            "*.tests10_plugins",
            "*.tests10_plugins.*",
            "tests10_plugins.*",
            "tests10_plugins",
        ]
    ),

    # packages = ['tofu','tofu.geom'],
    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    # py_modules=["my_module"],
    # List run-time dependencies here. These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "setuptools>=40.8.0, <64",
        "numpy",
        "scipy",
        # "scikit-sparse",
        # "scikit-umfpack",
        "matplotlib",
        "contourpy",
        "requests",
        "svg.path",
        "Polygon3",
        "cython>=0.26",
        "datastock>=0.0.54",
        "bsplines2d>=0.0.25",
        "spectrally>=0.0.9",
    ],
    python_requires=">=3.6",

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        "dev": [
            "check-manifest",
            "coverage",
            "pytest",
            "sphinx",
            "sphinx-gallery",
            "sphinx_bootstrap_theme",
        ]
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here. If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #    # If any package contains *.txt, *.rst or *.npz files, include them:
    #    '': ['*.txt', '*.rst', '*.npz'],
    #    # And include any *.csv files found in the 'ITER' package, too:
    #    'ITER': ['*.csv'],
    # },
    package_data={
        "tofu.tests.tests01_geom.test_data": [
            "*.py", "*.txt", ".svg", ".npz"
        ],
        "tofu.tests.tests04_spectro.test_data": ["*.npz"],
        "tofu.tests.tests06_mesh.test_data": ['*.txt', '*.npz'],
        "tofu.geom.inputs": ["*.txt"],
        "tofu.spectro": ["*.txt"],
        "tofu.physics_tools.runaways.emission": ['*.csv'],
        "tofu.physics_tools.transmission.inputs_filter": ['*.txt', '*.csv'],
        "tofu.mag.mag_ripple": ['*.sh', '*.f']
    },
    include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html
    # installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # executable scripts can be declared here
    # They can be python or non-python scripts
    # scripts=[
    # ],

    # entry_points point to functions in the package
    # Theye are generally preferable over scripts because they provide
    # cross-platform support and allow pip to create the appropriate form
    # of executable for the target platform.
    entry_points={
        'console_scripts': [
            'tofuplot=tofu.entrypoints.tofuplot:main',
            'tofucalc=tofu.entrypoints.tofucalc:main',
            'tofu-version=scripts.tofuversion:main',
            'tofu-custom=scripts.tofucustom:main',
            'tofu=scripts.tofu_bash:main',
        ],
    },

    py_modules=['_updateversion'],

    # Extensions and commands
    ext_modules=extensions,
    cmdclass={"build_ext": build_ext,
              "clean": CleanCommand},
    include_dirs=[np.get_include()],
)
