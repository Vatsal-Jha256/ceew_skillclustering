import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from general paths 
text_similarity_df_path = './datasets/text_similarity_df.csv'
text_similarity_sector_pair_dfs_path = './datasets/text_similarity_sector_pair_dfs.pkl'
merged_df_job_path = './datasets/merged_df_job_context.csv'

# Load datasets
text_similarity_df = pd.read_csv(text_similarity_df_path)
text_similarity_sector_pair_dfs = pd.read_pickle(text_similarity_sector_pair_dfs_path)
merged_df_job = pd.read_csv(merged_df_job_path)

# Function to get top N similar jobs between two sectors
def text_similarity_close_jobs(sector1, sector2, top_n=10):
    if (sector1, sector2) in text_similarity_sector_pair_dfs.keys():
        similarity_df = text_similarity_sector_pair_dfs[(sector1, sector2)]
    elif (sector2, sector1) in text_similarity_sector_pair_dfs.keys():
        similarity_df = text_similarity_sector_pair_dfs[(sector2, sector1)]
    else:
        st.error("Sector keys not found")
        return None
    return similarity_df.head(top_n)

# Function to find similarity score between two specific jobs
def find_text_job_similarity(job1, job2):
    if job1 == job2:
        st.error("You cannot compare the same job with itself. Please select different jobs.")
        return None

    try:
        text_similarity_score = text_similarity_df[
            ((text_similarity_df['Job 1'] == job1) & (text_similarity_df['Job 2'] == job2)) |
            ((text_similarity_df['Job 1'] == job2) & (text_similarity_df['Job 2'] == job1))
        ]['Similarity Score'].values[0]
    except IndexError:
        st.error("One of the job roles is not found.")
        return None

    return text_similarity_score

# Function to find matching jobs in a specified sector
def matched_jobs_in_sector(job, sector, df=None, top_n=5):
    if df is None:
        df = text_similarity_df.copy()
    df = df[((df['Job 1'] == job) & (df['Sector 2'] == sector)) | 
            ((df['Job 2'] == job) & (df['Sector 1'] == sector))]

    if df.empty:
        st.error("No matching jobs found in the selected sector.")
        return

    condition = df['Job 2'] == job
    df.loc[condition, ['Sector 1', 'Sector 2']] = df.loc[condition, ['Sector 2', 'Sector 1']].values
    df.loc[condition, ['Job 1', 'Job 2']] = df.loc[condition, ['Job 2', 'Job 1']].values

    sorted_df = df.sort_values(by="Similarity Score", ascending=False)

    st.write(f"\nTop matching jobs for {job} in {sector}:")
    st.dataframe(sorted_df.head(top_n))

    fig_top = px.bar(sorted_df.head(top_n), x='Job 2', y='Similarity Score',
                     title=f'Top Matching Jobs for {job} in {sector}',
                     labels={'Job 2': 'Job', 'Similarity Score': 'Similarity Score'},
                     range_y=[0, 1])  # Fixing the range of y-axis to avoid values >1
    st.plotly_chart(fig_top)

    st.write(f"\nLeast matching jobs for {job} in {sector}:")
    st.dataframe(sorted_df.tail(top_n))

    fig_least = px.bar(sorted_df.tail(top_n), x='Job 2', y='Similarity Score',
                       title=f'Least Matching Jobs for {job} in {sector}',
                       labels={'Job 2': 'Job', 'Similarity Score': 'Similarity Score'},
                       range_y=[0, 1])  # Fixing the range of y-axis to avoid values >1
    st.plotly_chart(fig_least)

# Function to calculate average similarity between sectors
def average_similarity_between_sectors():
    text_average_similarity = text_similarity_df.groupby(['Sector 1', 'Sector 2'])['Similarity Score'].mean().reset_index()
    text_average_similarity = text_average_similarity.sort_values(by='Similarity Score', ascending=False).reset_index(drop=True)
    
    st.write(text_average_similarity)
    
    pivot_df = text_average_similarity.pivot(index='Sector 1', columns='Sector 2', values='Similarity Score')
    fig = px.imshow(pivot_df, text_auto=True, aspect="auto",
                    title='Average Similarity Between Sectors',
                    labels={'color': 'Similarity Score'})
    st.plotly_chart(fig)

# Streamlit app interface with tabs for analysis and methodology
st.title("Job Similarity Analysis")

# Tabs
tab1, tab2 = st.tabs(["Analysis", "Methodology"])

with tab1:
    # Section 1: Job-to-Job Similarity
    with st.expander("Find Similarity Between Two Jobs in Different Sectors"):
        sector1 = st.selectbox('Select Sector for First Job', merged_df_job['Sector'].unique(), key='sector1_job1')
        job1_list = merged_df_job[merged_df_job['Sector'] == sector1]['QP/Job Role Name'].unique()
        job1 = st.selectbox('Select First Job', job1_list, key='job1')

        sector2 = st.selectbox('Select Sector for Second Job', merged_df_job['Sector'].unique(), key='sector2_job2')
        job2_list = merged_df_job[merged_df_job['Sector'] == sector2]['QP/Job Role Name'].unique()
        job2 = st.selectbox('Select Second Job', job2_list, key='job2')

        if st.button('Find Job Similarity'):
            similarity_score = find_text_job_similarity(job1, job2)
            if similarity_score is not None:
                st.write(f"The similarity score between '{job1}' and '{job2}' is: {similarity_score}")

    # Section 2: Sector-to-Sector Job Similarity
    with st.expander("Find Top Matching Jobs Between Two Sectors"):
        sector1 = st.selectbox('Select First Sector', merged_df_job['Sector'].unique(), key='sector1_top')
        sector2 = st.selectbox('Select Second Sector', merged_df_job['Sector'].unique(), key='sector2_top')

        if st.button('Find Top Matching Jobs', key='top_matching_jobs'):
            top_jobs_df = text_similarity_close_jobs(sector1, sector2)
            if top_jobs_df is not None:
                st.dataframe(top_jobs_df)

    # Section 3 & 4: Matching Jobs in Specified Sector
    with st.expander("Find Matching Jobs in a Specified Sector"):
        selected_sector = st.selectbox('Select Sector', merged_df_job['Sector'].unique(), key='sector3')

        job_list = merged_df_job[merged_df_job['Sector'] == selected_sector]['QP/Job Role Name'].unique()
        job = st.selectbox('Select Job', job_list, key='job3')

        target_sector = st.selectbox('Select Sector to Match With', merged_df_job['Sector'].unique(), key='sector_target')

        if st.button('Find Matching Jobs in Specified Sector'):
            matched_jobs_in_sector(job, target_sector)

    # Section 5: Sector Similarity Analysis
    with st.expander("View Average Similarity Between Sectors"):
        if st.button('Show Average Sector Similarity'):
            average_similarity_between_sectors()

with tab2:
    st.header("Methodology")
    st.write("This section provides insights into the distribution of similarity scores.")

    # Violin Plot
    st.subheader("Violin Plot of Similarity Scores")
    fig_violin, ax = plt.subplots()
    sns.violinplot(x=text_similarity_df['Similarity Score'], ax=ax)
    st.pyplot(fig_violin)

    # Box Plot (without outliers)
    st.subheader("Box Plot of Similarity Scores (Without Outliers)")
    fig_box, ax = plt.subplots()
    sns.boxplot(x=text_similarity_df['Similarity Score'], showfliers=False, ax=ax)
    st.pyplot(fig_box)

    # KDE Plot for Same vs Different Sector Similarity Scores
    st.subheader("KDE Plot for Similarity Scores (Same Sector vs Different Sector)")
    fig_kde, ax = plt.subplots()
    sns.kdeplot(text_similarity_df.loc[(text_similarity_df['Sector 1'] == text_similarity_df['Sector 2']), 'Similarity Score'],
                color='b', fill=True, label='Similarity Score - Same sector', ax=ax)
    sns.kdeplot(text_similarity_df.loc[(text_similarity_df['Sector 1'] != text_similarity_df['Sector 2']), 'Similarity Score'],
                color='r', fill=True, label='Similarity Score - Different sector', ax=ax)
    ax.legend()
    st.pyplot(fig_kde)
