"""
A gallery of Fusion Machines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This tutorial functions as a gallery of fusion machines that can easily be
loaded with `tofu`.
"""

###############################################################################
# We start by importing `tofu`.

import tofu as tf

###############################################################################
# `tofu` provides a geometry helper function that allows creating a
# configuration with a single call.
#
# Calling with empty arguments results in a default configuration. At the time
# of writing, this is ITER.
# By printing the `config` object, a text representation of its components is
# printed. This allows inspecting
# component names, number of sections, color or visibility.

config = tf.geom.utils.create_config()  # create default configuration
print(config)

###############################################################################
# To get a list of all available built-in configs, one has to know some details
# about `tofu`. Configurations can be accessed by names (ITER, WEST, JET, etc).

print(tf.geom.utils._DCONFIG_TABLE.keys())

###############################################################################
# With that being said, let's create a gallery of the "top 3" fusion machines
# provided by `tofu` to accelerate diagnostic development.

for fusion_machine in ['ITER', 'WEST', 'JET', 'NSTX']:
    config = tf.geom.utils.create_config(fusion_machine)
    config.plot()
