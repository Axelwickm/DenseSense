
class Person(object):
    incremental = 0

    def __init__(self, bounds, S, I, U, V):
        self.id = incremental
        incremental += 1

        self.bounds = bounds
        self.S = S
        self.I = I
        self.U = U
        self.V = V

    def become(self, otherInstance):
        self.incremental = otherInstance.incremental
        return self

