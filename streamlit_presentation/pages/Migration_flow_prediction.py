import streamlit as st
import pydeck as pdk
import numpy as np
from utils import show_code, countryCODE
import pandas as pd

# asyl_apps = pd.read_csv('../../data/asylum_monthly_clean.csv'
#                      # The following arguments are essential to correctly read the data frame
#                      ,index_col=0
#                      ,header=[0,1]) # take the first to rows as headers/ horizontal multiindex

country_list = list(countryCODE.keys())
country_list.insert(0, "")

st.title("Forecasting dashboard")
st.markdown("#### Enter a migration flow")

col1, col2 = st.columns(2)

with col1:
    from_op = st.selectbox(
        "from:",
        country_list,
    )

with col2:
    to_op = st.selectbox(
        "to:",
        country_list,
    )

if from_op != to_op and from_op != "" and to_op != "":
    st.pydeck_chart(
        pdk.Deck(
            map_style="dark",
            initial_view_state={
                "latitude": 35,
                "longitude": -5,
                "zoom": 1,
                "pitch": 90,
                "bearing": 0,
            },  # type: ignore
            layers=pdk.Layer(
                "ArcLayer",
                data=pd.read_json("./data/ve_de.json"),
                get_source_position=["lon", "lat"],
                get_target_position=["lon2", "lat2"],
                get_source_color="source_color",
                get_target_color="target_color",
                auto_highlight=True,
                width_scale=0.0001,
                get_width="width",
                width_min_pixels=3,
                width_max_pixels=30,
            ),
        )
    )
    dummy_pred = np.random.randint(300, 500)
    st.markdown(f"# {dummy_pred}")
    st.markdown(
        f"""#### Around {dummy_pred} asylum applications are to be expected in the comming 4 weeks."""
    )
    st.markdown(f"""###### This is 1 from 7072 ElasticNet ML models trained""")
