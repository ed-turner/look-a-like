import warnings

from .core.model import LALGBRegressor, LALGBClassifier

try:
    from .spark.model import LALGBSparkRegressor, LALGBSparkBinaryClassifier, LALGBSparkMultiClassifier

    warnings.warn("To use the spark package, you must install OpenBLAS adn Lapack, otherwise it won't run.")
except ImportError:
    warnings.warn("The spark package was not installed in this installation.. If you want to have spark implementation"
                  ", please install the package lal[spark]")
