
import streamlit as st
import pandas as pd
import json
import plotly.express as px
from pathlib import Path

# Set page config for a wider layout
st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    """Loads and preprocesses data from the JSON analysis files."""
    data_files = {
        'Normal': 'evaluation/results/accuracy_analysis_normal.json',
        'Scaled': 'evaluation/results/accuracy_analysis_all.json'
    }
    
    all_records = []
    for dataset_name, file_path in data_files.items():
        path = Path(file_path)
        if path.exists():
            with open(path, 'r') as f:
                records = json.load(f)['records']
                for r in records:
                    r['dataset'] = dataset_name
                all_records.extend(records)

    df = pd.DataFrame(all_records)

    # Filter out "Could not find answer within step limit"
    initial_rows = len(df)
    df = df[df['answer'] != "Could not find answer within step limit."]
    st.sidebar.info(f"Filtered {initial_rows - len(df)} 'No Answer' records.")

    def simplify_sys(name):
        """Simplifies system names for cleaner plotting."""
        if 'Hybrid-RAG' in name or 'RAG(' in name:
            return 'RAG'
        if 'Agentic' in name:
            if 'navigational' in name: return 'Agentic (Nav)'
            if 'simplified' in name: return 'Agentic (Simp)'
            if 'explicit' in name: return 'Agentic (Expl)'
            return 'Agentic'
        return name
    
    df['system_type'] = df['system'].apply(simplify_sys)
    return df

# --- Main App ---
st.title("🔬 SemaTree vs. RAG: Performance Dashboard")

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Dashboard Filters")
selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    options=df['dataset'].unique(),
    index=0,
    key="dataset_selector"
)

all_categories = sorted(df['category'].unique())
selected_categories = st.sidebar.multiselect(
    "Select Question Categories",
    options=all_categories,
    default=all_categories,
    key="category_selector"
)

# Filter data based on selections
filtered_df = df[
    (df['dataset'] == selected_dataset) &
    (df['category'].isin(selected_categories))
].copy()

st.header(f"Category-wise Analysis: {selected_dataset} Dataset")

if not filtered_df.empty:
    # --- FIX: Aggregate data before plotting ---
    agg_df = filtered_df.groupby(['category', 'system_type']).agg(
        semantic_similarity=('semantic_similarity', 'mean'),
        keyword_recall=('keyword_recall', 'mean'),
        latency_ms=('latency_ms', 'mean') # Also average latency
    ).reset_index()

    # --- Category-wise Difference Plots ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Semantic Similarity (Accuracy)")
        fig_sim = px.bar(
            agg_df,  # Use aggregated data
            x='category',
            y='semantic_similarity',
            color='system_type',
            barmode='group',
            title=f"Average Semantic Similarity by Category",
            labels={'semantic_similarity': 'Avg. Similarity Score', 'category': 'Question Category', 'system_type': 'System'},
            height=500
        )
        fig_sim.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_sim, use_container_width=True)

    with col2:
        st.subheader("Keyword Recall")
        fig_key = px.bar(
            agg_df,  # Use aggregated data
            x='category',
            y='keyword_recall',
            color='system_type',
            barmode='group',
            title=f"Average Keyword Recall by Category",
            labels={'keyword_recall': 'Avg. Keyword Recall Score', 'category': 'Question Category', 'system_type': 'System'},
            height=500
        )
        fig_key.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_key, use_container_width=True)

    # --- Latency vs. Accuracy ---
    st.header("Latency vs. Accuracy Trade-off")
    # For scatter plot of averages
    fig_scatter = px.scatter(
        agg_df,
        x='latency_ms',
        y='semantic_similarity',
        color='system_type',
        # symbol='category', removed as categories are already x-axis in bar charts
        title='Average Latency vs. Semantic Similarity',
        labels={'latency_ms': 'Average Latency (ms)', 'semantic_similarity': 'Average Semantic Similarity'},
        hover_data=['category'] # Now hover to see category
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


    # --- Raw Data Explorer ---
    with st.expander("Explore Aggregated & Raw Data"):
        st.subheader("Aggregated Averages")
        st.dataframe(agg_df)
        st.subheader("Raw Records")
        st.dataframe(filtered_df)

else:
    st.warning("No data available for the selected filters. Please adjust your selections.")

st.sidebar.info(
    "This dashboard visualizes the average performance of different retrieval systems. "
    "Use the filters to compare SemaTree variants against RAG."
)
