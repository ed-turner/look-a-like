import warnings

from .core.model import LALGBRegressor, LALGBClassifier

try:
    from .spark.model import LALGBSparkRegressor, LALGBSparkBinaryClassifier, LALGBSparkMultiClassifier
except ImportError:
    warnings.warn("The spark package was not installed in this installation.. If you want to have spark implementation"
                  ", please install the package lal[spark]")
