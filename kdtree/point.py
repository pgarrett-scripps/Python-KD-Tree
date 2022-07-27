class Point(list):
    def __new__(cls, value, name=None, values=None):
        s = super(Point, cls).__new__(cls, value)
        return s

    @staticmethod
    def create(values, data):
        point = Point(values)
        point.data = data
        return point
