"""
Creating a geometry with Inkscape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a tutorial that shows you how to draw a geometry in Inkscape
and load it in tofu.
"""

###############################################################################
# To see a tutorial on how to draw a vessel's geometry on Inkscape
# by tracing an image from a PDF file for example, and save it to a `svg` file,
# to load it and use it in tofu, you can check the video below. Basically, you
# just need to use straight *Bezier curves* to draw the closed polygons that
# will define the vessel and optionally the structures. To define a PFC
# structure, just add a fill color.
# You can also draw a line to automatically scale the configuration to a
# known measure.

###############################################################################

###############################################################################
# .. raw:: html
#
#    <div class="text-center">
#    <iframe width="560" height="315"
#    src="https://www.youtube.com/embed/MFwMZL7JjhI"
#    title="YouTube video player"
#    frameborder="0" allow="accelerometer; autoplay; clipboard-write;
#    encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
#    </iframe>
#    </div>

###############################################################################
# Now, on how to load it and use it in tofu (also shown in the video). We start
# by importing `tofu`.

import tofu as tf

###############################################################################
# `tofu` provides a geometry helper function that allows creating a
# configuration from a `svg` file. Supposing you saved it in a file named
# `"myfirstgeom.svg"`, do
config = tf.geom.Config.from_svg("myfirstgeom.svg", Name="Test", Exp="Test")

###############################################################################
# It already gives you some information on what you loaded, but to see what it
# actually contains:
print(config)

###############################################################################
# To plot it, simply do
config.plot()

###############################################################################
# We can see that the z- and r-axis might not be exactly what we wanted. We
# might also want to adjust the scale. Tofu let's fix those parameters
# when loading the configuration.
config = tf.geom.Config.from_svg("myfirstgeom.svg", Name="Test", Exp="Test",
                                 z0=-140,
                                 r0=10,
                                 scale=0.5
                                 )
config.plot()

###############################################################################
# Or, even better, if you have a figure where there is known measure you can
# let tofu scale the figure automatically for you
config = tf.geom.Config.from_svg(
    "from_pdf.svg",
    Name="Traced from pdf",
    Exp="Test",
    res=10,
    point_ref1=(0.7, -2),
    point_ref2=(2.8, 2)
)
config.plot()
# sphinx_gallery_thumbnail_number = 3
