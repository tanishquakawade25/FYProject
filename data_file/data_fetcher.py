import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def stock_data():
    st.title("📶 Analysis")

    # Sidebar input for stock tickers and date range
    st.sidebar.header("Add Stocks to Analyze")
    manual_stocks = st.sidebar.text_input(
        "Enter Stock Tickers (comma-separated, e.g., RELIANCE.NS,TCS.NS):", ""
    )
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-07-18"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-11-10"))

    # Extract stock tickers
    stocks_to_analyze = [ticker.strip() for ticker in manual_stocks.split(",") if ticker.strip()]

    # Button to fetch data
    if st.sidebar.button("Fetch Data"):
        if start_date >= end_date:
            st.error("Start date must be before the end date!")
        elif not stocks_to_analyze:
            st.warning("Please enter at least one stock ticker.")
        else:
            # Fetch stock data
            stock_data = pd.DataFrame()
            for ticker in stocks_to_analyze:
                try:
                    data = yf.Ticker(ticker).history(start=start_date, end=end_date)
                    if not data.empty:
                        stock_data[ticker] = data['Close']
                    else:
                        st.warning(f"No data found for ticker: {ticker}")
                except Exception as e:
                    st.warning(f"Could not fetch data for ticker: {ticker}. Error: {e}")

            # Fetch VIX data
            try:
                vix_data = yf.Ticker("^VIX").history(start=start_date, end=end_date)
                vix_data = vix_data[['Close']].rename(columns={'Close': 'VIX Index'})
            except Exception as e:
                st.error(f"Could not fetch VIX data. Error: {e}")
                vix_data = pd.DataFrame()

            if not stock_data.empty and not vix_data.empty:
                # Align VIX data with stock data by date
                stock_data['Date'] = stock_data.index.date
                vix_data['Date'] = vix_data.index.date
                stock_data.reset_index(drop=True, inplace=True)
                vix_data.reset_index(drop=True, inplace=True)

                # Merge the stock data with VIX data
                merged_data = pd.merge(stock_data, vix_data, on="Date", how="inner")

                view_option= st.radio(
                    "Choose Mode of Visualization:",
                    ("Tabular", "Graphical"),
                    horizontal=True,
                )

                if view_option == "Tabular":
                    # Show the merged data
                    st.write("Merged Data:")
                    st.dataframe(merged_data)

                    # Allow downloading the merged data as CSV
                    csv_data = merged_data.to_csv(index=False)
                    st.download_button(
                        label="Download Merged Data",
                        data=csv_data,
                        file_name="merged_stock_data.csv",
                        mime="text/csv",
                        icon="⬇️",
                    )

                elif view_option == "Graphical":
                    # Check if merged_data is not empty and contains required columns
                    if "VIX Index" not in merged_data.columns or merged_data.shape[1] <= 2:
                        st.error("Not enough data to plot. Ensure you have stock data and VIX Index.")
                    else:
                        # Convert 'Date' column to datetime if necessary
                        merged_data["Date"] = pd.to_datetime(merged_data["Date"])

                        # Plot the graph using Matplotlib
                        fig, ax1 = plt.subplots(figsize=(12, 6))

                        # Plot stock data on the primary Y-axis
                        for column in merged_data.columns:
                            if column not in ["Date", "VIX Index"]:
                                ax1.plot(merged_data["Date"], merged_data[column], label=column)
                                ax1.set_xlabel("Date", fontsize=12)
                                ax1.set_ylabel("Stock Prices", fontsize=12)
                                ax1.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

                                # Customize the x-axis with a specific interval (e.g., every 15 days)
                                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=15))  # Tick every 15 days
                                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # Format as 'YYYY-MM-DD'
                                plt.xticks(rotation=45)

                                # Plot VIX data on the secondary Y-axis
                                ax2 = ax1.twinx()
                                ax2.plot(merged_data["Date"], merged_data["VIX Index"], label="VIX Index", color="red", linewidth=2)
                                ax2.set_ylabel("VIX Index", fontsize=12, color="red")
                                ax2.tick_params(axis='y', labelcolor="red")

                                # Combine legends
                                lines1, labels1 = ax1.get_legend_handles_labels()
                                lines2, labels2 = ax2.get_legend_handles_labels()
                                ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

                                # Show the plot in Streamlit
                                st.pyplot(fig)

                                # Save the figure as an image
                                img_path = "stock_analysis.png"

                                # Save the figure
                                fig.savefig(img_path)

                                # Allow users to download the plot image
                                with open(img_path, "rb") as img_file:
                                    st.download_button(
                                        label="Download stock_analysis Image",
                                        data=img_file,
                                        file_name="stock_analysis.png",
                                        mime="image/png",
                                        icon="📥",
                                    )
                
            else:
                st.error("No valid data available to display.")

def app():
    stock_data()