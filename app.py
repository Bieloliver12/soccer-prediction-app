import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
import os
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Soccer Team Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-bottom: 1rem;
    }
    .team-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data_path = "E0.csv"
    
    # Try multiple possible paths
    possible_paths = [
        "E0.csv",
        "Club-Football-Match-Data-2000-2025/data/E0.csv",
        "data/E0.csv",
        "./E0.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    
    if df is None:
        # Create sample data if file not found
        st.warning("Data file not found. Using sample data.")
        np.random.seed(42)
        teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man United', 'Man City', 'Tottenham', 'Leicester', 'West Ham', 'Everton', 'Newcastle']
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        data = []
        for _ in range(500):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            home_goals = np.random.poisson(1.5)
            away_goals = np.random.poisson(1.2)
            
            data.append({
                'Date': np.random.choice(dates),
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'FTHG': home_goals,
                'FTAG': away_goals,
                'FTR': 'H' if home_goals > away_goals else ('A' if away_goals > home_goals else 'D')
            })
        
        df = pd.DataFrame(data)
    
    # Add calculated columns
    df['Result'] = df.apply(lambda x: 'Home Win' if x['FTHG'] > x['FTAG'] 
                           else ('Away Win' if x['FTHG'] < x['FTAG'] else 'Draw'), axis=1)
    df['Total_Goals'] = df['FTHG'] + df['FTAG']
    df['Goal_Difference'] = df['FTHG'] - df['FTAG']
    
    # Convert Date if exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    return df

def get_team_data(df, team_name):
    
    team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()
    
    # Create team-specific columns
    team_matches['Team_Goals'] = np.where(team_matches['HomeTeam'] == team_name, 
                                         team_matches['FTHG'], team_matches['FTAG'])
    team_matches['Opponent_Goals'] = np.where(team_matches['HomeTeam'] == team_name, 
                                             team_matches['FTAG'], team_matches['FTHG'])
    team_matches['Venue'] = np.where(team_matches['HomeTeam'] == team_name, 'Home', 'Away')
    team_matches['Opponent'] = np.where(team_matches['HomeTeam'] == team_name, 
                                       team_matches['AwayTeam'], team_matches['HomeTeam'])
    
    # Team result from team's perspective
    team_matches['Team_Result'] = np.where(team_matches['Team_Goals'] > team_matches['Opponent_Goals'], 'Win',
                                          np.where(team_matches['Team_Goals'] < team_matches['Opponent_Goals'], 'Loss', 'Draw'))
    
    return team_matches

def page_data_overview():
    
    st.markdown('<h1 class="main-header">⚽ Premier League Data Overview</h1>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        st.error("Unable to load data.")
        return
    
    # Dataset Overview
    st.markdown('<h2 class="sub-header">📊 Dataset Summary</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", len(df))
    with col2:
        total_teams = len(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))
        st.metric("Total Teams", total_teams)
    with col3:
        if 'Date' in df.columns:
            seasons = len(df['Date'].dt.year.unique())
            st.metric("Seasons", seasons)
        else:
            st.metric("Seasons", "N/A")
    with col4:
        st.metric("Data Points", len(df.columns))
    
    # League Overview Stats
    st.markdown('<h2 class="sub-header">🏆 League Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall match results
        result_counts = df['Result'].value_counts()
        fig_results = px.pie(values=result_counts.values, names=result_counts.index,
                            title="Overall Match Results Distribution",
                            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        st.plotly_chart(fig_results, use_container_width=True)
    
    with col2:
        # Goals per match distribution
        fig_goals = px.histogram(df, x='Total_Goals', nbins=10,
                               title="Goals per Match Distribution",
                               color_discrete_sequence=['#96CEB4'])
        fig_goals.update_layout(showlegend=False)
        st.plotly_chart(fig_goals, use_container_width=True)
    
    # Top performers
    st.markdown('<h2 class="sub-header">⭐ Top Performers</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most wins overall
        home_wins = df[df['Result'] == 'Home Win']['HomeTeam'].value_counts()
        away_wins = df[df['Result'] == 'Away Win']['AwayTeam'].value_counts()
        total_wins = (home_wins.fillna(0) + away_wins.fillna(0)).sort_values(ascending=False).head(10)
        
        fig_wins = px.bar(x=total_wins.index, y=total_wins.values,
                         title="Teams with Most Wins",
                         color_discrete_sequence=['#FFD93D'])
        fig_wins.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_wins, use_container_width=True)
    
    with col2:
        # Highest scoring teams
        home_goals = df.groupby('HomeTeam')['FTHG'].sum()
        away_goals = df.groupby('AwayTeam')['FTAG'].sum()
        total_goals = (home_goals.fillna(0) + away_goals.fillna(0)).sort_values(ascending=False).head(10)
        
        fig_goals_team = px.bar(x=total_goals.index, y=total_goals.values,
                               title="Highest Scoring Teams",
                               color_discrete_sequence=['#FF6B6B'])
        fig_goals_team.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_goals_team, use_container_width=True)
    
    # Sample data preview
    st.markdown('<h2 class="sub-header">📋 Recent Matches</h2>', unsafe_allow_html=True)
    
    # Show recent matches
    display_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result']
    available_cols = [col for col in display_cols if col in df.columns]
    
    if 'Date' in df.columns:
        recent_matches = df.sort_values('Date', ascending=False).head(10)
    else:
        recent_matches = df.head(10)
    
    st.dataframe(recent_matches[available_cols], use_container_width=True)

def page_team_eda():
    
    st.markdown('<h1 class="main-header">🏆 Team Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        return
    
    # Team selection
    all_teams = sorted(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_team = st.selectbox("🎯 Select Team for Analysis:", all_teams)
    
    with col2:
        st.markdown(f'<div class="team-card"><h3>{selected_team}</h3><p>Team Analytics</p></div>', 
                   unsafe_allow_html=True)
    
    if not selected_team:
        st.warning("Please select a team to view analytics.")
        return
    
    # Get team data
    team_data = get_team_data(df, selected_team)
    
    if len(team_data) == 0:
        st.error(f"No matches found for {selected_team}")
        return
    
    # Team performance metrics
    st.markdown('<h2 class="sub-header">📊 Team Performance Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_matches = len(team_data)
    wins = len(team_data[team_data['Team_Result'] == 'Win'])
    draws = len(team_data[team_data['Team_Result'] == 'Draw'])
    losses = len(team_data[team_data['Team_Result'] == 'Loss'])
    goals_scored = team_data['Team_Goals'].sum()
    goals_conceded = team_data['Opponent_Goals'].sum()
    
    with col1:
        st.metric("Total Matches", total_matches)
    with col2:
        st.metric("Wins", wins)
    with col3:
        st.metric("Draws", draws)
    with col4:
        st.metric("Losses", losses)
    with col5:
        win_rate = (wins / total_matches * 100) if total_matches > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Goals Scored", goals_scored)
    with col2:
        st.metric("Goals Conceded", goals_conceded)
    with col3:
        goal_diff = goals_scored - goals_conceded
        st.metric("Goal Difference", f"{int(goal_diff):+.0f}")
    with col4:
        avg_goals = goals_scored / total_matches if total_matches > 0 else 0
        st.metric("Goals/Match", f"{avg_goals:.1f}")
    
    # Performance charts
    st.markdown('<h2 class="sub-header">📈 Performance Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Win/Loss/Draw distribution
        result_counts = team_data['Team_Result'].value_counts()
        fig_results = px.pie(values=result_counts.values, names=result_counts.index,
                           title=f"{selected_team} - Match Results",
                           color_discrete_map={'Win': '#28a745', 'Draw': '#ffc107', 'Loss': '#dc3545'})
        st.plotly_chart(fig_results, use_container_width=True)
    
    with col2:
        # Home vs Away performance
        venue_performance = team_data.groupby(['Venue', 'Team_Result']).size().unstack(fill_value=0)
        fig_venue = px.bar(venue_performance, 
                          title=f"{selected_team} - Home vs Away Performance",
                          color_discrete_map={'Win': '#28a745', 'Draw': '#ffc107', 'Loss': '#dc3545'})
        st.plotly_chart(fig_venue, use_container_width=True)
    
    # Goals analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Goals scored distribution
        fig_goals_scored = px.histogram(team_data, x='Team_Goals', nbins=8,
                                       title=f"{selected_team} - Goals Scored Distribution",
                                       color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig_goals_scored, use_container_width=True)
    
    with col2:
        # Goals scored vs conceded
        fig_scatter = px.scatter(team_data, x='Team_Goals', y='Opponent_Goals',
                               color='Team_Result',
                               title=f"{selected_team} - Goals Scored vs Conceded",
                               color_discrete_map={'Win': '#28a745', 'Draw': '#ffc107', 'Loss': '#dc3545'})
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Performance over time (if date available)
    if 'Date' in team_data.columns and not team_data['Date'].isna().all():
        st.markdown('<h2 class="sub-header">📅 Performance Over Time</h2>', unsafe_allow_html=True)
        
        # Sort by date and create cumulative stats
        team_data_sorted = team_data.sort_values('Date').reset_index(drop=True)
        team_data_sorted['Cumulative_Points'] = team_data_sorted['Team_Result'].map(
            {'Win': 3, 'Draw': 1, 'Loss': 0}).cumsum()
        team_data_sorted['Match_Number'] = range(1, len(team_data_sorted) + 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Points accumulation over time
            fig_points = px.line(team_data_sorted, x='Match_Number', y='Cumulative_Points',
                               title=f"{selected_team} - Points Accumulation",
                               markers=True)
            st.plotly_chart(fig_points, use_container_width=True)
        
        with col2:
            # Goals per match over time
            team_data_sorted['Rolling_Avg_Goals'] = team_data_sorted['Team_Goals'].rolling(window=5, min_periods=1).mean()
            fig_rolling = px.line(team_data_sorted, x='Match_Number', y='Rolling_Avg_Goals',
                                title=f"{selected_team} - Rolling Average Goals (5 matches)",
                                markers=True)
            st.plotly_chart(fig_rolling, use_container_width=True)
    
    # Head-to-head against opponents
    st.markdown('<h2 class="sub-header">🆚 Performance Against Opponents</h2>', unsafe_allow_html=True)
    
    opponent_stats = team_data.groupby('Opponent').agg({
        'Team_Result': lambda x: (x == 'Win').sum(),
        'Team_Goals': 'mean',
        'Opponent_Goals': 'mean'
    }).round(2)
    
    opponent_stats.columns = ['Wins', 'Avg_Goals_Scored', 'Avg_Goals_Conceded']
    opponent_stats = opponent_stats.sort_values('Wins', ascending=False).head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_h2h = px.bar(opponent_stats, x=opponent_stats.index, y='Wins',
                        title=f"{selected_team} - Most Wins Against Opponents",
                        color_discrete_sequence=['#ff7f0e'])
        fig_h2h.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_h2h, use_container_width=True)
    
    with col2:
        st.markdown("**Top Matchups:**")
        for idx, (opponent, row) in enumerate(opponent_stats.head(5).iterrows(), 1):
            st.write(f"{idx}. vs {opponent}: {row['Wins']} wins, {row['Avg_Goals_Scored']:.1f} goals avg")
    
    # Recent form
    st.markdown('<h2 class="sub-header">🔥 Recent Form</h2>', unsafe_allow_html=True)
    
    if 'Date' in team_data.columns and not team_data['Date'].isna().all():
        recent_matches = team_data.sort_values('Date', ascending=False).head(10)
    else:
        recent_matches = team_data.tail(10)
    
    # Create a visual representation of recent form
    recent_results = recent_matches['Team_Result'].tolist()
    form_colors = {'Win': '🟢', 'Draw': '🟡', 'Loss': '🔴'}
    form_string = ' '.join([form_colors[result] for result in reversed(recent_results)])
    
    st.markdown(f"**Last 10 matches:** {form_string}")
    
    # Recent matches table
    display_cols = ['Date', 'Opponent', 'Venue', 'Team_Goals', 'Opponent_Goals', 'Team_Result']
    available_cols = [col for col in display_cols if col in recent_matches.columns]
    
    st.dataframe(recent_matches[available_cols].reset_index(drop=True), use_container_width=True)

def page_team_statistics():
    
    st.markdown('<h1 class="main-header">📈 Team Statistical Analysis</h1>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        return
    
    # Team selection
    all_teams = sorted(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_team = st.selectbox("📊 Select Team for Statistical Analysis:", all_teams)
    
    with col2:
        # Add comparison team option
        compare_team = st.selectbox("🆚 Compare with (optional):", ['None'] + all_teams)
    
    if not selected_team:
        return
    
    team_data = get_team_data(df, selected_team)
    
    # Detailed Statistics
    st.markdown('<h2 class="sub-header">📊 Detailed Team Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🏠 Home Performance**")
        home_data = team_data[team_data['Venue'] == 'Home']
        if len(home_data) > 0:
            home_wins = len(home_data[home_data['Team_Result'] == 'Win'])
            home_total = len(home_data)
            st.metric("Home Win Rate", f"{home_wins/home_total*100:.1f}%")
            st.metric("Home Goals/Match", f"{home_data['Team_Goals'].mean():.1f}")
            st.metric("Home Goals Conceded", f"{home_data['Opponent_Goals'].mean():.1f}")
    
    with col2:
        st.markdown("**🚌 Away Performance**")
        away_data = team_data[team_data['Venue'] == 'Away']
        if len(away_data) > 0:
            away_wins = len(away_data[away_data['Team_Result'] == 'Win'])
            away_total = len(away_data)
            st.metric("Away Win Rate", f"{away_wins/away_total*100:.1f}%")
            st.metric("Away Goals/Match", f"{away_data['Team_Goals'].mean():.1f}")
            st.metric("Away Goals Conceded", f"{away_data['Opponent_Goals'].mean():.1f}")
    
    with col3:
        st.markdown("**📈 Overall Stats**")
        st.metric("Total Matches", len(team_data))
        st.metric("Points Per Match", f"{(len(team_data[team_data['Team_Result'] == 'Win']) * 3 + len(team_data[team_data['Team_Result'] == 'Draw'])) / len(team_data):.2f}")
        st.metric("Goal Difference", f"{team_data['Team_Goals'].sum() - team_data['Opponent_Goals'].sum():+.0f}")
    
    # Statistical distributions
    st.markdown('<h2 class="sub-header">📊 Statistical Distributions</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Goals scored distribution with statistics
        fig_dist = px.histogram(team_data, x='Team_Goals', 
                               title=f"{selected_team} - Goals Distribution",
                               marginal="box")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Statistics
        st.markdown("**Goals Scored Statistics:**")
        st.write(f"Mean: {team_data['Team_Goals'].mean():.2f}")
        st.write(f"Median: {team_data['Team_Goals'].median():.2f}")
        st.write(f"Std Dev: {team_data['Team_Goals'].std():.2f}")
        st.write(f"Max: {team_data['Team_Goals'].max()}")
        st.write(f"Min: {team_data['Team_Goals'].min()}")
    
    with col2:
        # Goals conceded distribution
        fig_conceded = px.histogram(team_data, x='Opponent_Goals',
                                   title=f"{selected_team} - Goals Conceded Distribution",
                                   marginal="box")
        st.plotly_chart(fig_conceded, use_container_width=True)
        
        # Statistics
        st.markdown("**Goals Conceded Statistics:**")
        st.write(f"Mean: {team_data['Opponent_Goals'].mean():.2f}")
        st.write(f"Median: {team_data['Opponent_Goals'].median():.2f}")
        st.write(f"Std Dev: {team_data['Opponent_Goals'].std():.2f}")
        st.write(f"Max: {team_data['Opponent_Goals'].max()}")
        st.write(f"Min: {team_data['Opponent_Goals'].min()}")
    
    # Performance patterns
    st.markdown('<h2 class="sub-header">🎯 Performance Patterns</h2>', unsafe_allow_html=True)
    
    # Goals correlation analysis
    correlation = team_data['Team_Goals'].corr(team_data['Opponent_Goals'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scoring patterns
        high_scoring = len(team_data[team_data['Team_Goals'] >= 3])
        clean_sheets = len(team_data[team_data['Opponent_Goals'] == 0])
        
        patterns_data = pd.DataFrame({
            'Pattern': ['High Scoring (3+ goals)', 'Clean Sheets', 'Failed to Score', 'Conceded 3+'],
            'Count': [
                high_scoring,
                clean_sheets,
                len(team_data[team_data['Team_Goals'] == 0]),
                len(team_data[team_data['Opponent_Goals'] >= 3])
            ],
            'Percentage': [
                high_scoring/len(team_data)*100,
                clean_sheets/len(team_data)*100,
                len(team_data[team_data['Team_Goals'] == 0])/len(team_data)*100,
                len(team_data[team_data['Opponent_Goals'] >= 3])/len(team_data)*100
            ]
        })
        
        fig_patterns = px.bar(patterns_data, x='Pattern', y='Percentage',
                             title=f"{selected_team} - Performance Patterns",
                             color_discrete_sequence=['#2E86C1'])
        fig_patterns.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_patterns, use_container_width=True)
    
    with col2:
        # Score line frequency
        team_data['Score_Line'] = team_data['Team_Goals'].astype(str) + '-' + team_data['Opponent_Goals'].astype(str)
        score_freq = team_data['Score_Line'].value_counts().head(10)
        
        fig_scores = px.bar(x=score_freq.index, y=score_freq.values,
                           title=f"{selected_team} - Most Common Scorelines",
                           color_discrete_sequence=['#E74C3C'])
        st.plotly_chart(fig_scores, use_container_width=True)
    
    # Comparison with another team (if selected)
    if compare_team != 'None':
        st.markdown(f'<h2 class="sub-header">🆚 {selected_team} vs {compare_team} Comparison</h2>', 
                   unsafe_allow_html=True)
        
        compare_data = get_team_data(df, compare_team)
        
        # Create comparison metrics
        comparison_df = pd.DataFrame({
            'Metric': ['Matches Played', 'Win Rate (%)', 'Goals/Match', 'Conceded/Match', 'Points/Match'],
            selected_team: [
                len(team_data),
                len(team_data[team_data['Team_Result'] == 'Win'])/len(team_data)*100,
                team_data['Team_Goals'].mean(),
                team_data['Opponent_Goals'].mean(),
                (len(team_data[team_data['Team_Result'] == 'Win']) * 3 + len(team_data[team_data['Team_Result'] == 'Draw'])) / len(team_data)
            ],
            compare_team: [
                len(compare_data),
                len(compare_data[compare_data['Team_Result'] == 'Win'])/len(compare_data)*100,
                compare_data['Team_Goals'].mean(),
                compare_data['Opponent_Goals'].mean(),
                (len(compare_data[compare_data['Team_Result'] == 'Win']) * 3 + len(compare_data[compare_data['Team_Result'] == 'Draw'])) / len(compare_data)
            ]
        })
        
        st.dataframe(comparison_df.round(2), use_container_width=True)
        
        # Head-to-head record
        h2h_matches = df[((df['HomeTeam'] == selected_team) & (df['AwayTeam'] == compare_team)) |
                        ((df['HomeTeam'] == compare_team) & (df['AwayTeam'] == selected_team))]
        
        if len(h2h_matches) > 0:
            st.markdown("**Head-to-Head Record:**")
            
            team1_wins = len(h2h_matches[((h2h_matches['HomeTeam'] == selected_team) & (h2h_matches['FTHG'] > h2h_matches['FTAG'])) |
                                       ((h2h_matches['AwayTeam'] == selected_team) & (h2h_matches['FTAG'] > h2h_matches['FTHG']))])
            team2_wins = len(h2h_matches[((h2h_matches['HomeTeam'] == compare_team) & (h2h_matches['FTHG'] > h2h_matches['FTAG'])) |
                                       ((h2h_matches['AwayTeam'] == compare_team) & (h2h_matches['FTAG'] > h2h_matches['FTHG']))])
            draws = len(h2h_matches[h2h_matches['FTHG'] == h2h_matches['FTAG']])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{selected_team} Wins", team1_wins)
            with col2:
                st.metric("Draws", draws)
            with col3:
                st.metric(f"{compare_team} Wins", team2_wins)
        else:
            st.info("No head-to-head matches found between these teams.")

def page_team_insights():
    
    st.markdown('<h1 class="main-header"> Team ML Insights</h1>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        return
    
    # Team selection
    all_teams = sorted(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_team = st.selectbox("🎯 Select Team for ML Analysis:", all_teams)
    
    with col2:
        analysis_type = st.selectbox("📊 Analysis Type:", [
            "Performance Prediction",
            "Win Probability Analysis"
        ])
    
    if not selected_team:
        return
    
    team_data = get_team_data(df, selected_team)
    
    # Prepare ML features
    def prepare_ml_features(df):
        
        # Encode teams
        le_home = LabelEncoder()
        le_away = LabelEncoder()
        
        df_ml = df.copy()
        df_ml['HomeTeam_Encoded'] = le_home.fit_transform(df_ml['HomeTeam'])
        df_ml['AwayTeam_Encoded'] = le_away.fit_transform(df_ml['AwayTeam'])
        
        return df_ml, le_home, le_away
    
    df_ml, le_home, le_away = prepare_ml_features(df)
    
    if analysis_type == "Performance Prediction":
        st.markdown('<h2 class="sub-header">🎯 Performance Prediction Analysis</h2>', unsafe_allow_html=True)
        
        # Train a model to predict team performance
        X = df_ml[['HomeTeam_Encoded', 'AwayTeam_Encoded']]
        y = df_ml['Result']
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict team's performance against all other teams
        team_encoded = le_home.transform([selected_team])[0] if selected_team in le_home.classes_ else 0
        
        predictions = {}
        for opponent in all_teams:
            if opponent != selected_team:
                opp_encoded = le_away.transform([opponent])[0] if opponent in le_away.classes_ else 0
                
                # Home match prediction
                home_pred = model.predict_proba([[team_encoded, opp_encoded]])[0]
                home_probs = dict(zip(model.classes_, home_pred))
                
                # Away match prediction  
                away_pred = model.predict_proba([[opp_encoded, team_encoded]])[0]
                away_probs = dict(zip(model.classes_, away_pred))
                
                predictions[opponent] = {
                    'Home_Win_Prob': home_probs.get('Home Win', 0),
                    'Away_Win_Prob': away_probs.get('Away Win', 0),
                    'Avg_Win_Prob': (home_probs.get('Home Win', 0) + away_probs.get('Away Win', 0)) / 2
                }
        
        if predictions:
            # Create predictions dataframe
            pred_df = pd.DataFrame.from_dict(predictions, orient='index')
            pred_df = pred_df.sort_values('Avg_Win_Prob', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Best matchups
                st.markdown(f"**Best Matchups for {selected_team}:**")
                fig_best = px.bar(pred_df.head(8), x=pred_df.head(8).index, y='Avg_Win_Prob',
                                 title=f"Highest Win Probability vs Opponents",
                                 color_discrete_sequence=['#28a745'])
                fig_best.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_best, use_container_width=True)
            
            with col2:
                # Challenging matchups
                st.markdown(f"**Most Challenging Matchups:**")
                fig_worst = px.bar(pred_df.tail(8), x=pred_df.tail(8).index, y='Avg_Win_Prob',
                                  title=f"Lowest Win Probability vs Opponents",
                                  color_discrete_sequence=['#dc3545'])
                fig_worst.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_worst, use_container_width=True)
            
            # Top 5 and Bottom 5
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🟢 Easiest Opponents:**")
                for i, (opp, data) in enumerate(pred_df.head(5).iterrows(), 1):
                    st.write(f"{i}. {opp}: {data['Avg_Win_Prob']:.1%} win probability")
            
            with col2:
                st.markdown("**🔴 Toughest Opponents:**")
                for i, (opp, data) in enumerate(pred_df.tail(5).iterrows(), 1):
                    st.write(f"{i}. {opp}: {data['Avg_Win_Prob']:.1%} win probability")
    
    elif analysis_type == "Win Probability Analysis":
        st.markdown('<h2 class="sub-header">📊 Win Probability Analysis</h2>', unsafe_allow_html=True)
        
        # Analyze factors affecting win probability
        team_data['Win'] = (team_data['Team_Result'] == 'Win').astype(int)
        
        # Calculate win probability by various factors
        col1, col2 = st.columns(2)
        
        with col1:
            # Win probability by venue
            venue_wins = team_data.groupby('Venue')['Win'].agg(['count', 'sum', 'mean']).round(3)
            venue_wins['Win_Rate'] = venue_wins['mean'] * 100
            
            fig_venue = px.bar(x=venue_wins.index, y=venue_wins['Win_Rate'],
                              title=f"{selected_team} - Win Rate by Venue",
                              color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig_venue, use_container_width=True)
            
            st.markdown("**Venue Performance:**")
            for venue, data in venue_wins.iterrows():
                st.write(f"• {venue}: {data['Win_Rate']:.1f}% ({int(data['sum'])}/{int(data['count'])})")
        
        with col2:
            # Win probability by goals scored
            goals_wins = team_data.groupby('Team_Goals')['Win'].agg(['count', 'sum', 'mean']).round(3)
            goals_wins = goals_wins[goals_wins['count'] >= 3]  # Only include goals with 3+ occurrences
            
            fig_goals = px.bar(x=goals_wins.index, y=goals_wins['mean'] * 100,
                              title=f"{selected_team} - Win Rate by Goals Scored",
                              color_discrete_sequence=['#2ca02c'])
            st.plotly_chart(fig_goals, use_container_width=True)
            
            st.markdown("**Goals Impact:**")
            for goals, data in goals_wins.iterrows():
                st.write(f"• {int(goals)} goals: {data['mean']*100:.1f}% win rate")
        
        # Logistic regression for win probability
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features
        features = ['Team_Goals', 'Opponent_Goals']
        available_features = [f for f in features if f in team_data.columns]
        
        if len(available_features) >= 2:
            X = team_data[available_features]
            y = team_data['Win']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train logistic regression
            log_reg = LogisticRegression()
            log_reg.fit(X_scaled, y)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': available_features,
                'Importance': abs(log_reg.coef_[0])
            }).sort_values('Importance', ascending=False)
            
            st.markdown("**🎯 Factors Affecting Win Probability:**")
            fig_importance = px.bar(feature_importance, x='Feature', y='Importance',
                                   title="Feature Importance for Winning")
            st.plotly_chart(fig_importance, use_container_width=True)

def page_prediction():
    
    st.markdown('<h1 class="main-header">🎮 Match Prediction</h1>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        return
    
    # Prepare and train model automatically
    @st.cache_resource
    def prepare_prediction_model(df):
        
        df_ml = df.copy()
        
        # Encode teams
        le_home = LabelEncoder()
        le_away = LabelEncoder()
        
        df_ml['HomeTeam_Encoded'] = le_home.fit_transform(df_ml['HomeTeam'])
        df_ml['AwayTeam_Encoded'] = le_away.fit_transform(df_ml['AwayTeam'])
        
        # Prepare features
        feature_cols = ['HomeTeam_Encoded', 'AwayTeam_Encoded']
        
        # Add additional features if available
        additional_features = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC']
        for feature in additional_features:
            if feature in df_ml.columns:
                feature_cols.append(feature)
        
        X = df_ml[feature_cols].fillna(df_ml[feature_cols].mean())
        y = df_ml['Result']
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, le_home, le_away, feature_cols, accuracy
    
    # Load model
    with st.spinner("Preparing prediction model..."):
        model, le_home, le_away, feature_cols, accuracy = prepare_prediction_model(df)
    
    st.success(f"✅ Prediction model ready! Accuracy: {accuracy:.1%}")
    
    st.markdown('<h2 class="sub-header">⚽ Predict Match Outcome</h2>', unsafe_allow_html=True)
    
    # Get unique teams
    all_teams = sorted(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏠 Home Team**")
        home_team = st.selectbox("Select Home Team:", all_teams, key="home")
        
        if home_team:
            home_data = get_team_data(df, home_team)
            home_home_data = home_data[home_data['Venue'] == 'Home']
            
            if len(home_home_data) > 0:
                home_win_rate = len(home_home_data[home_home_data['Team_Result'] == 'Win']) / len(home_home_data) * 100
                home_goals_avg = home_home_data['Team_Goals'].mean()
                
                st.metric("Home Win Rate", f"{home_win_rate:.1f}%")
                st.metric("Home Goals/Game", f"{home_goals_avg:.1f}")
    
    with col2:
        st.markdown("**🚌 Away Team**")
        away_team = st.selectbox("Select Away Team:", all_teams, key="away")
        
        if away_team:
            away_data = get_team_data(df, away_team)
            away_away_data = away_data[away_data['Venue'] == 'Away']
            
            if len(away_away_data) > 0:
                away_win_rate = len(away_away_data[away_away_data['Team_Result'] == 'Win']) / len(away_away_data) * 100
                away_goals_avg = away_away_data['Team_Goals'].mean()
                
                st.metric("Away Win Rate", f"{away_win_rate:.1f}%")
                st.metric("Away Goals/Game", f"{away_goals_avg:.1f}")
    
    # Prediction button
    if st.button("🎯 Predict Match Outcome", type="primary", use_container_width=True):
        if home_team == away_team:
            st.error(" Please select different teams for home and away.")
        else:
            # Encode teams
            home_encoded = le_home.transform([home_team])[0]
            away_encoded = le_away.transform([away_team])[0]
            
            # Prepare features
            features = [home_encoded, away_encoded]
            
            # Add additional features if model was trained with them
            additional_features = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC']
            for feature in additional_features:
                if feature in df.columns and feature in feature_cols:
                    # Use team averages for additional features
                    home_avg = df[df['HomeTeam'] == home_team][feature].mean()
                    away_avg = df[df['AwayTeam'] == away_team][feature].mean()
                    
                    if pd.isna(home_avg):
                        home_avg = df[feature].mean()
                    if pd.isna(away_avg):
                        away_avg = df[feature].mean()
                    
                    features.extend([home_avg, away_avg])
            
            # Ensure we have the right number of features
            while len(features) < len(feature_cols):
                features.append(0)
            
            features_array = np.array(features[:len(feature_cols)]).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features_array)[0]
            probabilities = model.predict_proba(features_array)[0]
            
            # Display results
            st.markdown('<h3 class="sub-header">🏆 Prediction Results</h3>', unsafe_allow_html=True)
            
            # Main prediction
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 15px; color: white; margin: 1rem 0;">
                <h2>{home_team} vs {away_team}</h2>
                <h1>🏆 {prediction}</h1>
                <h3>Confidence: {max(probabilities):.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability breakdown
            classes = model.classes_
            prob_dict = dict(zip(classes, probabilities))
            
            st.markdown("**📊 Outcome Probabilities:**")
            
            cols = st.columns(len(classes))
            for i, (outcome, prob) in enumerate(prob_dict.items()):
                with cols[i]:
                    if outcome == 'Home Win':
                        color = "🟢"
                    elif outcome == 'Away Win':
                        color = "🔴"
                    else:
                        color = "🟡"
                    
                    st.metric(f"{color} {outcome}", f"{prob:.1%}")
                    st.progress(prob)
            
            # Detailed probability chart
            fig_probs = px.bar(x=list(prob_dict.keys()), y=list(prob_dict.values()),
                              title="Match Outcome Probabilities",
                              color=list(prob_dict.values()),
                              color_continuous_scale='viridis')
            fig_probs.update_layout(showlegend=False, yaxis_title="Probability")
            st.plotly_chart(fig_probs, use_container_width=True)
            
            # Historical head-to-head
            st.markdown('<h3 class="sub-header">📊 Historical Head-to-Head</h3>', unsafe_allow_html=True)
            
            h2h = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                    ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))]
            
            if len(h2h) > 0:
                # H2H statistics
                home_wins_h2h = len(h2h[(h2h['HomeTeam'] == home_team) & (h2h['FTHG'] > h2h['FTAG'])])
                away_wins_h2h = len(h2h[(h2h['AwayTeam'] == home_team) & (h2h['FTAG'] > h2h['FTHG'])])
                draws_h2h = len(h2h[h2h['FTHG'] == h2h['FTAG']])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{home_team} Wins", home_wins_h2h)
                with col2:
                    st.metric("Draws", draws_h2h)
                with col3:
                    st.metric(f"{away_team} Wins", away_wins_h2h)
                
                # Recent H2H matches
                st.markdown("**Recent H2H Matches:**")
                recent_h2h = h2h.tail(5)[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result']]
                st.dataframe(recent_h2h, use_container_width=True)
                
            else:
                st.info(" No historical matches found between these teams.")
            
            # Additional insights
            st.markdown('<h3 class="sub-header">🔍 Additional Insights</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{home_team} Recent Form:**")
                home_recent = get_team_data(df, home_team).tail(5)
                home_recent_results = home_recent['Team_Result'].tolist()
                form_colors = {'Win': '🟢', 'Draw': '🟡', 'Loss': '🔴'}
                home_form = ' '.join([form_colors[result] for result in reversed(home_recent_results)])
                st.write(f"Last 5: {home_form}")
                
                home_recent_points = len([r for r in home_recent_results if r == 'Win']) * 3 + len([r for r in home_recent_results if r == 'Draw'])
                st.write(f"Points from last 5: {home_recent_points}/15")
            
            with col2:
                st.markdown(f"**{away_team} Recent Form:**")
                away_recent = get_team_data(df, away_team).tail(5)
                away_recent_results = away_recent['Team_Result'].tolist()
                away_form = ' '.join([form_colors[result] for result in reversed(away_recent_results)])
                st.write(f"Last 5: {away_form}")
                
                away_recent_points = len([r for r in away_recent_results if r == 'Win']) * 3 + len([r for r in away_recent_results if r == 'Draw'])
                st.write(f"Points from last 5: {away_recent_points}/15")

def main():
    
    st.sidebar.title("⚽ Soccer Team Analytics")
    st.sidebar.markdown("---")
    
    pages = {
        " Data Overview": page_data_overview,
        " Team EDA": page_team_eda,
        " Team Statistics": page_team_statistics,
        " Team ML Insights": page_team_insights,
        " Match Prediction": page_prediction
    }
    
    selected_page = st.sidebar.selectbox("Choose Analysis:", list(pages.keys()))
    
    # Run the selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()