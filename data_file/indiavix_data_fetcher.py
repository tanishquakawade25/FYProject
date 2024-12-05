import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Function to fetch India VIX data
@st.cache_data(ttl=30)  # Cache data for 30 seconds
def fetch_indiavix_data():
    try:
        vix = yf.Ticker("^INDIAVIX")
        data = vix.history(period="1d", interval="1m")  # Fetch 1-minute interval data
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to display India VIX data
def display_vix_data():
    # Fetch data
    with st.spinner("Fetching live data..."):
        data = fetch_indiavix_data()

    if not data.empty:
        # Extract key metrics
        current_price = data['Close'].iloc[-1]
        open_price = data['Open'].iloc[0]
        high_price = data['High'].max()
        low_price = data['Low'].min()
        previous_close = data['Close'].iloc[-2]

        # Display key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Current Price", f"{current_price:.2f}")
        col2.metric("Open Price", f"{open_price:.2f}")
        col3.metric("High Price", f"{high_price:.2f}")
        col4.metric("Low Price", f"{low_price:.2f}")
        col5.metric("Previous Close", f"{previous_close:.2f}")

        # Plot real-time graph using Plotly
        st.subheader("Live India VIX Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='red')))
        fig.update_layout(
            title="India VIX Intraday Trend",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Allow user to download the stock data as a CSV file
        csv_data = data.to_csv(index=True)  # Include the index (timestamp) in the CSV
        st.download_button(
            label="Download india vix Data as CSV",
            data=csv_data,
            file_name="india_vix_data.csv",
            mime="text/csv",
            icon="📄",
        )

    else:
        st.error("No data available. Please try refreshing.")





# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import plotly.graph_objects as go

# # Function to fetch India VIX data
# @st.cache_data(ttl=30)  # Cache data for 30 seconds
# def fetch_indiavix_data():
#     try:
#         vix = yf.Ticker("^INDIAVIX")
#         data = vix.history(period="1d", interval="1m")  # Fetch 1-minute interval data
#         return data
#     except Exception as e:
#         st.error(f"Error fetching data: {e}")
#         return pd.DataFrame()

# # Streamlit app configuration
# st.set_page_config(page_title="India VIX Live Dashboard", layout="wide")

# # Title and description
# st.title("India VIX Live Dashboard")
# st.markdown("""
# ### Real-Time India VIX Trend
# Manually refresh the data by clicking the "Refresh Data" button.
# """)

# def display_vix_data():
#     # Fetch data
#     with st.spinner("Fetching live data..."):
#         data = fetch_indiavix_data()

#     if not data.empty:
#         # Extract key metrics
#         current_price = data['Close'].iloc[-1]
#         open_price = data['Open'].iloc[0]
#         high_price = data['High'].max()
#         low_price = data['Low'].min()
#         previous_close = data['Close'].iloc[-2]

#         # Display key metrics
#         col1, col2, col3, col4, col5 = st.columns(5)
#         col1.metric("Current Price", f"{current_price:.2f}")
#         col2.metric("Open Price", f"{open_price:.2f}")
#         col3.metric("High Price", f"{high_price:.2f}")
#         col4.metric("Low Price", f"{low_price:.2f}")
#         col5.metric("Previous Close", f"{previous_close:.2f}")

#         # Plot real-time graph using Plotly
#         st.subheader("Live India VIX Trend")
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='red')))
#         fig.update_layout(
#             title="India VIX Intraday Trend",
#             xaxis_title="Time",
#             yaxis_title="Price",
#             xaxis_rangeslider_visible=True,
#             template="plotly_dark"
#         )
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.error("No data available. Please try refreshing.")

# # Check if the module is being run directly
# if __name__ == "__main__":
#     display_vix_data()








# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import plotly.graph_objects as go

# # Function to fetch India VIX data
# @st.cache_data(ttl=30)  # Cache data for 30 seconds
# def fetch_indiavix_data():
#     try:
#         vix = yf.Ticker("^INDIAVIX")
#         data = vix.history(period="1d", interval="1m")  # Fetch 1-minute interval data
#         return data
#     except Exception as e:
#         st.error(f"Error fetching data: {e}")
#         return pd.DataFrame()

# # Streamlit app configuration
# st.set_page_config(page_title="India VIX Live Dashboard", layout="wide")

# # Title and description
# st.title("India VIX Live Dashboard")
# st.markdown("""
# ### Real-Time India VIX Trend
# Manually refresh the data by clicking the "Refresh Data" button.
# """)

# # # Sidebar settings
# # st.sidebar.header("Settings")
# # if st.sidebar.button("Refresh Data"):
# #     st.experimental_rerun()  # Allows the user to refresh manually if needed

# # Fetch data
# with st.spinner("Fetching live data..."):
#     data = fetch_indiavix_data()

# if not data.empty:
#     # Extract key metrics
#     current_price = data['Close'].iloc[-1]
#     open_price = data['Open'].iloc[0]
#     high_price = data['High'].max()
#     low_price = data['Low'].min()
#     previous_close = data['Close'].iloc[-2]

#     # Display key metrics
#     col1, col2, col3, col4, col5 = st.columns(5)
#     col1.metric("Current Price", f"{current_price:.2f}")
#     col2.metric("Open Price", f"{open_price:.2f}")
#     col3.metric("High Price", f"{high_price:.2f}")
#     col4.metric("Low Price", f"{low_price:.2f}")
#     col5.metric("Previous Close", f"{previous_close:.2f}")

#     # Plot real-time graph using Plotly
#     st.subheader("Live India VIX Trend")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='red')))
#     fig.update_layout(
#         title="India VIX Intraday Trend",
#         xaxis_title="Time",
#         yaxis_title="Price",
#         xaxis_rangeslider_visible=True,
#         template="plotly_dark"
#     )
#     st.plotly_chart(fig, use_container_width=True)
# else:
#     st.error("No data available. Please try refreshing.")