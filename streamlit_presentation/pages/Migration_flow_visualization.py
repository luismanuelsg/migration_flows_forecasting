# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import inspect
import textwrap
import pandas as pd
import pydeck as pdk
from utils import show_code, countryCODE


from urllib.error import URLError


def mapping_demo():
    @st.cache_data
    def from_data_file(filename):
        url = (
            "http://raw.githubusercontent.com/streamlit/"
            "example-data/master/hello/v1/%s" % filename
        )
        return pd.read_json(url)

    try:
        ALL_LAYERS = {
            "VE->DE": pdk.Layer(
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
            "SD->DE": pdk.Layer(
                "ArcLayer",
                data=pd.read_json("./data/sd_de.json"),
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
            "Input variables flow": pdk.Layer(
                "ArcLayer",
                data=pd.read_json("./data/input_variables.json"),
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
            "Input variables sources": pdk.Layer(
                type="IconLayer",
                data=pd.read_json("icons.json"),
                get_icon="icon_data",
                get_size=4,
                pickable=True,
                size_scale=15,
                get_position=["lon", "lat"],
            ),
        }
        st.sidebar.markdown("### Select layer")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, False)
        ]
        if selected_layers:
            st.pydeck_chart(
                pdk.Deck(
                    map_style="dark",
                    initial_view_state={
                        "latitude": 35,
                        "longitude": -5,
                        "zoom": 1,
                        "pitch": 90,
                        "bearing": 90,
                    },  # type: ignore
                    layers=selected_layers,
                )
            )
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**
            Connection error: %s
        """
            % e.reason
        )


st.set_page_config(page_title="Migration flow", page_icon="🌍")
st.markdown("# Visualizing migration flows")
st.sidebar.header("Map layers")
st.write()
mapping_demo()
st.markdown("#### Input variables that the model monitors:")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("##### A. In origin/transit country")
    with st.expander("1. Labeled and classified socio-economical and political events"):
        st.image("./media/gdelt_logo.jpg")
    with st.expander("2. Google searches that indicate migration intention"):
        st.image("./media/google_trends.webp")
with col2:
    st.markdown("##### B. In between")
    with st.expander("3. Border crossing statistics"):
        st.image("./media/frontex-logo.png")
with col3:
    st.markdown("##### C. In destination country")
    with st.expander("4. Asylum application data and recognition rates"):
        st.image("./media/eurostat.png")

# show_code(mapping_demo)
