from .algorithm_specs import describe


class AlgorithmEntry:
    name = "lbfgsb_tv"

    def describe(self, variant=None):
        return describe(variant)


def list_algorithms():
    return ["lbfgsb_tv"]


def get_algorithm_entry(name):
    if name != "lbfgsb_tv":
        raise KeyError(name)
    return AlgorithmEntry()


def get_algorithm(name):
    get_algorithm_entry(name)
    from ..algorithms.lbfgsb_tv import LBFGSBTV

    return LBFGSBTV()
