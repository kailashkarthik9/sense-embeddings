import numpy

from scipy.spatial import distance


class StatisticsComputer:
    """
    This class encapsulates all the logic required to perform statistical analysis including computation and
    visualization.
    """

    @staticmethod
    def get_vector_distance(v1: numpy.ndarray, v2: numpy.ndarray):
        return distance.minkowski(v1, v2)

    @staticmethod
    def get_vector_mean(vector: numpy.ndarray):
        return numpy.mean(vector)

    @staticmethod
    def get_vector_max(vector: numpy.ndarray):
        return numpy.max(vector)

    @staticmethod
    def get_vector_min(vector: numpy.ndarray):
        return numpy.min(vector)

    @staticmethod
    def get_vector_range(vector: numpy.ndarray):
        return numpy.max(vector) - numpy.min(vector)

    @staticmethod
    def get_vector_std_dev(vector: numpy.ndarray):
        return numpy.std(vector)
