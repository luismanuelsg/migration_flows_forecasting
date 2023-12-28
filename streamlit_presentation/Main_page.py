import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    # st.set_page_config(
    #     page_title="Hello",
    #     page_icon="👋",
    # )

    # st.write("# Forescasting migration flows")

    st.markdown(
        """
        # Forescasting asylum-related migration flows
        #### Humanitarian appliances of machine learning and data at scale
        """
        #     **👈 Select a demo from the sidebar** to see some examples
        #     of what Streamlit can do!
        #     ### Want to learn more?
        #     - Check out [streamlit.io](https://streamlit.io)
        #     - Jump into our [documentation](https://docs.streamlit.io)
        #     - Ask a question in our [community
        #       forums](https://discuss.streamlit.io)
        #     ### See more complex demos
        #     - Use a neural net to [analyze the Udacity Self-driving Car Image
        #       Dataset](https://github.com/streamlit/demo-self-driving)
        #     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
        # """
    )
    st.image("./media/mediterranean.jpg")
    st.markdown(
        """
        <div align="right"><h4>Luis Manuel Sandia Goncalves</h4></div>                  
        <div align="right"><h6>May 2023, Spiced Academy - Berlin</h6></div>
        """,
        unsafe_allow_html=True,
    )
    # st.sidebar.success("Select a demo above.")

    # st.markdown(
    #     """
    #     Streamlit is an open-source app framework built specifically for
    #     Machine Learning and Data Science projects.
    #     **👈 Select a demo from the sidebar** to see some examples
    #     of what Streamlit can do!
    #     ### Want to learn more?
    #     - Check out [streamlit.io](https://streamlit.io)
    #     - Jump into our [documentation](https://docs.streamlit.io)
    #     - Ask a question in our [community
    #       forums](https://discuss.streamlit.io)
    #     ### See more complex demos
    #     - Use a neural net to [analyze the Udacity Self-driving Car Image
    #       Dataset](https://github.com/streamlit/demo-self-driving)
    #     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    # """
    # )


if __name__ == "__main__":
    run()
