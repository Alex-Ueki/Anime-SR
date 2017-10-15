""" Tests for Modules/genomics.py. Currently only tests the Organism class
    because it's hard to mock tests for all the closures and random
    mutation functions
"""

from Modules.genomics import Organism

def test_organism():
    """ Test Organism class """

    org = Organism('ecoli')
    assert list(org) == ['ecoli', 0.0, 0, True]
    assert org.genome == 'ecoli' and org.fitness == 0.0 and org.epoch == 0 and org.improved

    org = Organism(['virus', -10.5])
    assert org.genome == 'virus' and org.fitness == -10.5 and org.epoch == 0 and org.improved
    assert list(org) == ['virus', -10.5, 0, True]

    org = Organism(['amoeba', 37.3, 20])
    assert list(org) == ['amoeba', 37.3, 20, True]
    assert org.genome == 'amoeba' and org.fitness == 37.3 and org.epoch == 20 and org.improved

    org = Organism(['hominid', -99, 123, False])
    assert list(org) == ['hominid', -99.0, 123, False]
    assert org.genome == 'hominid' and org.fitness == -99.0 and org.epoch == 123 and not org.improved

    assert repr(org) == str(list(org))
    assert str(org)
