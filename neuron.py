"""
Single computational unit that stores one numeric value.

Neurons are grouped into Layers; each neuron can have an activation
function applied to transform its stored value.
"""


class Neuron:

    def __init__(self, value=0):
        self.value = value

    def apply(self, function):
        """Apply an activation function in place."""
        self.value = function(self.value)

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value
