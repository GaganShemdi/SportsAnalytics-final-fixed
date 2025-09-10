import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# App Configurations
st.set_page_config(page_title="Enhanced Sports Analytics Dashboard", layout="wide")


# Load Data
@st.cache_data
def load_data(file_path=None):
    if file_path:
        data = pd.read_csv(r"C:\Users\Admin\Downloads\sports_data_data.csv")
    else:
        data = pd.read_csv(r"C:\Users\Admin\Downloads\sports_data_data.csv")
    return data


# Header Section
st.markdown(
    """
    <style>
    .main-title {
        font-size: 48px;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
    }
    .sub-title {
        font-size: 20px;
        color: #666666;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    <div class="main-title">Enhanced Sports Analytics Dashboard</div>
    <div class="sub-title">Explore team and player performance with advanced insights</div>
    """,
    unsafe_allow_html=True
)

# Upload Functionality
uploaded_file = st.file_uploader("Upload your sports data CSV file", type="csv")
data = load_data(uploaded_file) if uploaded_file else load_data()

if not data.empty:
    # Sidebar Filters
    st.sidebar.header("Filters")
    selected_team = st.sidebar.multiselect(
        "Select Team", options=data["Team"].unique(), default=data["Team"].unique()
    )
    selected_metric = st.sidebar.selectbox(
        "Select Metric to Analyze", options=["Points", "Assists", "Rebounds"]
    )
    min_points = st.sidebar.slider(
        "Minimum Points",
        min_value=int(data["Points"].min()),
        max_value=int(data["Points"].max()),
        value=int(data["Points"].min()),
    )

    # Filter Data
    filtered_data = data[(data["Team"].isin(selected_team)) & (data["Points"] >= min_points)]

    # Tabs for Sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Filtered Data", "Team Performance", "Top Performers", "Player Comparisons", "Pie Charts", "Predictions"
    ])

    # Filtered Data Tab
    with tab1:
        st.subheader("Filtered Data")
        if filtered_data.empty:
            st.warning("No data matches the current filters. Please adjust the filters.")
        else:
            st.dataframe(filtered_data)

    # Team Performance Tab
    with tab2:
        st.subheader("Team Performance")
        if not filtered_data.empty:
            team_stats = filtered_data.groupby("Team")[[selected_metric]].sum().reset_index()
            chart = alt.Chart(team_stats).mark_bar().encode(
                x=alt.X("Team", sort="-y"),
                y=alt.Y(selected_metric, title=f"Total {selected_metric}"),
                tooltip=["Team", selected_metric],
                color="Team"
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No team data available for the selected filters.")

    # Top Performers Tab
    with tab3:
        st.subheader("Top Performers")
        if not filtered_data.empty:
            top_performers = filtered_data.sort_values(by=selected_metric, ascending=False).head(5)
            st.table(top_performers[["Player", "Team", selected_metric]])
        else:
            st.info("No player data available for the selected filters.")

    # Player Comparisons Tab
    with tab4:
        st.subheader("Player Comparisons")
        selected_players = st.multiselect(
            "Select Players to Compare", options=filtered_data["Player"].unique()
        )
        if selected_players:
            comparison_data = filtered_data[filtered_data["Player"].isin(selected_players)]
            comparison_chart = alt.Chart(comparison_data).mark_bar().encode(
                x="Player",
                y=selected_metric,
                color="Team",
                tooltip=["Player", "Team", selected_metric]
            ).interactive()
            st.altair_chart(comparison_chart, use_container_width=True)
        else:
            st.info("Select players to see comparisons.")

    # Pie Charts Tab
    with tab5:
        st.subheader("Pie Chart: Team Contributions")
        if not filtered_data.empty:
            # Prepare Data
            team_contributions = filtered_data.groupby("Team")[selected_metric].sum().reset_index()
            team_contributions["Percentage"] = (
                    team_contributions[selected_metric] / team_contributions[selected_metric].sum() * 100
            )

            # Matplotlib Pie Chart
            st.write("*Matplotlib Pie Chart*")
            fig, ax = plt.subplots(figsize=(5, 5))
            wedges, texts, autotexts = ax.pie(
                team_contributions[selected_metric],
                labels=team_contributions["Team"],
                autopct=lambda pct: f"{pct:.1f}%\n({int(pct * team_contributions[selected_metric].sum() / 100)})",
                startangle=140,
                colors=plt.cm.tab20.colors
            )
            ax.set_title(f"{selected_metric} Contributions by Team")
            st.pyplot(fig)

            # Altair Pie Chart
            st.write("*Altair Pie Chart*")
            alt_chart = alt.Chart(team_contributions).mark_arc().encode(
                theta=alt.Theta(field=selected_metric, type="quantitative"),
                color=alt.Color(field="Team", type="nominal"),
                tooltip=["Team", f"{selected_metric}:Q", "Percentage:Q"]
            )
            st.altair_chart(alt_chart, use_container_width=True)
        else:
            st.info("No data available for the selected filters.")

    # Predictions Tab
    with tab6:
        st.subheader("Predict Future Performance")
        if not filtered_data.empty:
            X = filtered_data[["Assists", "Rebounds"]]
            y = filtered_data["Points"]
            model = LinearRegression()
            model.fit(X, y)
            filtered_data["Predicted Points"] = model.predict(X)
            st.dataframe(filtered_data[["Player", "Points", "Predicted Points"]])
        else:
            st.info("No data available for predictions.")
else:
    st.error("No data available. Please upload a valid CSV file or check the default dataset.")