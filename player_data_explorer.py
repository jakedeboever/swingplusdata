import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("final kfold data.csv")

    # Remove Player ID if exists
    if "Player ID" in df.columns:
        df = df.drop(columns=["Player ID"])

    # Reorder columns: ensure year first, then Team, age, then Side, Swing+
    cols = list(df.columns)
    new_order = []

    if "year" in cols:
        new_order.append("year")
    if "Team" in cols:
        new_order.append("Team")
    if "age" in cols:
        new_order.append("age")
    if "Side" in cols:
        new_order.append("Side")
    if "Swing+" in cols:
        new_order.append("Swing+")

    for c in cols:
        if c not in new_order:
            new_order.append(c)

    df = df[new_order]
    return df

df = load_data()

st.title("MLB Player & Team Data Explorer")

# --- Player Search Bar on Top ---
all_players = sorted(df["name"].unique())
player_search = st.text_input("Search Player")

if player_search:
    filtered_players = [p for p in all_players if player_search.lower() in p.lower()]
else:
    filtered_players = all_players

selected_players = st.multiselect("Select Players", filtered_players, default=filtered_players)

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Team filter (column is "Team")
all_teams = sorted(df["Team"].dropna().unique())
selected_teams = st.sidebar.multiselect("Select Teams", all_teams, default=all_teams)

# Aggregation mode
agg_mode = st.sidebar.radio("Aggregation Mode", ["Aggregate All Years (Player)", "Per Year (Player)", "Aggregate All Years (Team)"])

# --- Filter Data ---
filtered_df = df[df["name"].isin(selected_players) & df["Team"].isin(selected_teams)]

# --- Aggregate Data ---
if agg_mode == "Aggregate All Years (Player)":
    aggr_df = filtered_df.groupby("name", as_index=False).mean(numeric_only=True)
elif agg_mode == "Per Year (Player)":
    aggr_df = filtered_df.copy()
else:  # Aggregate All Years (Team)
    aggr_df = filtered_df.groupby("Team", as_index=False).mean(numeric_only=True)

# --- Stat Filters ---
num_cols = aggr_df.select_dtypes(include=['float64','int64']).columns.tolist()

st.sidebar.subheader("Stat Filters")
for col in num_cols:
    min_val = float(aggr_df[col].min())
    max_val = float(aggr_df[col].max())
    sel_min, sel_max = st.sidebar.slider(
        f"{col}", min_val, max_val, (min_val, max_val)
    )
    aggr_df = aggr_df[(aggr_df[col] >= sel_min) & (aggr_df[col] <= sel_max)]

st.subheader("Filtered Data")
st.dataframe(aggr_df)

# --- Download Data Button ---
st.download_button(
    label="Download Filtered Data as CSV",
    data=aggr_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_data.csv",
    mime="text/csv"
)

# --- Plotting ---
if len(num_cols) >= 2:
    x_axis = st.selectbox("X-axis", num_cols, index=0)
    y_axis = st.selectbox("Y-axis", num_cols, index=1)

    if x_axis and y_axis:
        st.subheader(f"Scatter Plot: {x_axis} vs {y_axis}")
        fig, ax = plt.subplots()
        if agg_mode == "Per Year (Player)" and "year" in aggr_df.columns:
            sns.scatterplot(data=aggr_df, x=x_axis, y=y_axis, hue="year", ax=ax)
        elif agg_mode == "Aggregate All Years (Team)":
            sns.scatterplot(data=aggr_df, x=x_axis, y=y_axis, hue="Team", ax=ax)
        else:
            sns.scatterplot(data=aggr_df, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)

        # Save plot to buffer for download
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        st.download_button(
            label="Download Scatter Plot as PNG",
            data=buf,
            file_name="scatter_plot.png",
            mime="image/png"
        )

        # Correlation
        corr = aggr_df[[x_axis, y_axis]].corr().iloc[0, 1]
        st.write(f"**Correlation Coefficient (r):** {corr:.3f}")
else:
    st.write("Not enough numeric columns to plot.")

