
class LowPassFilter(object):
    def __init__(self, a):
        self.a = a
        self.b = 1.0 - a
        self.last_val = 0.


    def get(self):
        return self.last_val


    def clear(self):
        self.last_val = 0.0


    def filter(self, val):
        val = self.a * val + self.b * self.last_val
        self.last_val = val
        return val
