class FExtractionGDELT:
    def __init__(self) -> None:
        pass


class FExtractionEUAsyl:
    def __init__(self) -> None:
        pass

    def get_dyad():
        pass

    def get_recognition_rate():
        pass


class FExtractionBordercrossing:
    def __init__(self) -> None:
        pass


class FEextractionGoogleTrends:
    def __init__(self) -> None:
        pass


class AdaENet(origin=None, destination=None):
    """Creates a model for a given dyad"""

    def __init__(self, origin, destination) -> None:
        pass


class MigrationFlowsVisualization:
    def __init__(self) -> None:
        pass


class dEarlyWarning:
    """
    Early warning object for a single dyad
    """

    def __init__(self, dyad_tuple):
        if isinstance(dyad_tuple, tuple):
            self.dyad = dyad_tuple
        else:
            raise TypeError("Dyad must be a tuple.")

    def set_dyad(self, dyad_tuple):
        if isinstance(dyad_tuple, tuple):
            self.dyad = dyad_tuple
        else:
            raise TypeError("Dyad must be a tuple.")

    def get_dyad(self):
        return self.dyad

    def show_dashboard():
        """
        Some seaborn plotting, or dataframe code
        """
        pass


class EarlyWarning(dEarlyWarning):
    """
    EarlyWarning object for the worldwide scope
    """


class MigrationFlowForecast(AdaENet):
    def __init__(self) -> None:
        pass
