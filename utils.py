# Copyright (C) 2013-2016 Martin Vejmelka, UC Denver

class Dict(dict):
    """
    A dictionary that allows member access to its keys.
    A convenience class.
    """

    def __init__(self, d):
        """
        Updates itself with d.
        """
        self.update(d)

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, item, value):
        self[item] = value
