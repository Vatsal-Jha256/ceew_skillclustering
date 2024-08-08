import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    try:
        text_similarity_score = text_similarity_df[
            ((text_similarity_df['Job 1'] == job1) & (text_similarity_df['Job 2'] == job2)) |
            ((text_similarity_df['Job 1'] == job2) & (text_similarity_df['Job 2'] == job1))
        ]['Similarity Score'].values[0]
    except IndexError:
        st.error("One of the job roles is not found.")
        return None

    return text_similarity_score

# Function to find matching jobs in other sectors
def matched_jobs_other_sectors(job, df=None, top_n=5):
    if df is None:
        df = text_similarity_df.copy()
    df = df[(df['Job 1'] == job) | (df['Job 2'] == job)]
    df = df[df['Job 1'] != df['Job 2']]
    if df.empty:
        st.error("Entered job not found")
        return
    st.success("Job found in dataframe. Proceeding with analysis")
    
    condition = df['Job 2'] == job
    df.loc[condition, ['Sector 1', 'Sector 2']] = df.loc[condition, ['Sector 2', 'Sector 1']].values
    df.loc[condition, ['NSQF 1', 'NSQF 2']] = df.loc[condition, ['NSQF 2', 'NSQF 1']].values
    df.loc[condition, ['Job 1', 'Job 2']] = df.loc[condition, ['Job 2', 'Job 1']].values
    
    sector_groups = df.groupby('Sector 2')
    avg_similarity = sector_groups['Similarity Score'].mean().reset_index()

    st.write(f"\nAvg similarity across sectors for {job}: \n")
    st.dataframe(avg_similarity.sort_values(by='Similarity Score', ascending=False))

    fig = px.bar(avg_similarity, x='Sector 2', y='Similarity Score',
                 title=f'Average Similarity for {job} across sectors',
                 labels={'Sector 2': 'Sector', 'Similarity Score': 'Average Similarity'})
    st.plotly_chart(fig)

    for sector, group in sector_groups:
        st.write(f"\nTop Match for {job} in sector: {sector}")
        st.dataframe(group[['Job 2', 'Sector 2', 'Similarity Score']].sort_values(by='Similarity Score', ascending=False).head(top_n))

        st.write(f"\nLeast Match for {job} in sector: {sector}")
        st.dataframe(group[['Job 2', 'Sector 2', 'Similarity Score']].sort_values(by='Similarity Score', ascending=True).head(top_n))

# Function to find top matching jobs across sectors
def matched_jobs(job, df=None):
    if df is None:
        df = text_similarity_df.copy()
    df = df[(df['Job 1'] == job) | (df['Job 2'] == job)]
    df = df[df['Job 1'] != df['Job 2']]
    if df.empty:
        st.error("Entered job not found")
        return
    st.success("Job found in dataframe. Proceeding with analysis")

    condition = df['Job 2'] == job
    df.loc[condition, ['Sector 1', 'Sector 2']] = df.loc[condition, ['Sector 2', 'Sector 1']].values
    df.loc[condition, ['NSQF 1', 'NSQF 2']] = df.loc[condition, ['NSQF 2', 'NSQF 1']].values
    df.loc[condition, ['Job 1', 'Job 2']] = df.loc[condition, ['Job 2', 'Job 1']].values
    
    sector = df.iloc[0]['Sector 1']
    df = df[df['Sector 2'] != sector]
    sorted_df = df.sort_values(by="Similarity Score", ascending=False)

    st.write(f"\nTop matching jobs for {job} across all sectors, except {sector}, are:")
    st.dataframe(sorted_df.head(10))

    fig = px.bar(sorted_df.head(10), x='Job 2', y='Similarity Score',
                 title=f'Top Matching Jobs for {job}',
                 labels={'Job 2': 'Job', 'Similarity Score': 'Similarity Score'})
    st.plotly_chart(fig)

# Streamlit app interface with expanders for each function
st.title("Job Similarity Analysis")

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

# Section 3: Matching Jobs Across Sectors
with st.expander("Find Top Matching Jobs Across Sectors"):
    selected_sector = st.selectbox('Select Sector', merged_df_job['Sector'].unique(), key='sector3')

    job_list = merged_df_job[merged_df_job['Sector'] == selected_sector]['QP/Job Role Name'].unique()
    job = st.selectbox('Select Job', job_list, key='job3')

    if st.button('Find Top Matching Jobs Across Sectors'):
        matched_jobs(job, text_similarity_df)

# Section 4: Sector-Wise Job Matching
with st.expander("Find Matching Jobs in Other Sectors"):
    selected_sector = st.selectbox('Select Sector', merged_df_job['Sector'].unique(), key='sector4')

    job_list = merged_df_job[merged_df_job['Sector'] == selected_sector]['QP/Job Role Name'].unique()
    job = st.selectbox('Select Job', job_list, key='job4')

    if st.button('Find Matching Jobs in Other Sectors'):
        matched_jobs_other_sectors(job, text_similarity_df)

# Section 5: Sector Similarity Analysis
with st.expander("View Average Similarity Between Sectors"):
    if st.button('Show Average Sector Similarity'):
        text_average_similarity = text_similarity_df.groupby(['Sector 1', 'Sector 2'])['Similarity Score'].mean().reset_index()
        text_average_similarity = text_average_similarity.sort_values(by='Similarity Score', ascending=False).reset_index(drop=True)
        
        st.write(text_average_similarity)
        
        pivot_df = text_average_similarity.pivot(index='Sector 1', columns='Sector 2', values='Similarity Score')
        fig = px.imshow(pivot_df, text_auto=True, aspect="auto",
                        title='Average Similarity Between Sectors',
                        labels={'color': 'Similarity Score'})
        st.plotly_chart(fig)
