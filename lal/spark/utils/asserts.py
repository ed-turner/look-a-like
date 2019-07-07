from lal.utils.asserts import AssertArgumentTypeBase

from pyspark.sql.dataframe import DataFrame


class AssertArgumentSparkDataFrame(AssertArgumentTypeBase):
    """

    """

    assert_type = DataFrame
