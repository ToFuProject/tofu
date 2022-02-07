

import matplotlib.ptyplot as plt


from ._DataCollection_class import DataCaollection


class AxesCollection(DataCollection):
    """ Handles matplotlib interactivity """

    _LPAXES = ['ax', 'type']

    def add_axes(self, key=None, ax=None):
        super.add_obj(which='axes', key=key, ax=ax)


    def add_mobile(self, key=None, handle=None):
        super.add_obj(which='mobile', key=key, handle=handle)

    @property
    def dax(self):
        return self.dobj['axes']


    def connect(self):
        pass
