import streamlit as st
import numpy as np
import pandas as pd
from utils import show_code, countryCODE, Eurostats_preprocessing, GDELT_preprocessing

# df = pd.read_csv('../data/asylum_monthly_clean.csv'
#                      # The following arguments are essential to correctly read the data frame
#                      ,index_col=0
#                      ,header=[0,1])

st.title("Forecasting migration flows requires big data preprocessing")
st.markdown("###### Around 300GB of CSV files had to be 'manually' preprocessed")
st.markdown("The code designed could potentially be used in a future python API")

with st.expander("GDELT Project"):
    st.video("./media/gdelt_landingpage.webm")

with st.expander(
    "Show example of 1 GDELT csv file for the globals events in 1 single day"
):
    st.video("./media/GDELT_eventsCSV_day.webm")

with st.expander("## GDELT Big data preprocessing code"):
    st.markdown(
        "###### _The GDELT preprocessing modules could potentially be implemented in a future GEDELT Python API_"
    )
    show_code(GDELT_preprocessing)

with st.expander("## Eurostats preprocessing code"):
    show_code(Eurostats_preprocessing)

with st.expander(
    "## Defining a common format to use: Migration flow datetime-dataframe"
):
    st.markdown(
        "#### Finding a standarized way _migration flows datetime DataFrames_ can be handled in pandas:"
    )
    st.markdown(
        """```python
migration_flow_df['from']['to']
    """
    )
    st.video("./media/Migration_flow_timeseries_format.webm")
# with st.expander("## Migration flow time series format"):
#
