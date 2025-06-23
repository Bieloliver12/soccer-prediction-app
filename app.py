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
import warnings
import os
warnings.filterwarnings('ignore')

# Simple page config
st.set_page_config(
    page_title="Soccer Analytics",
    page_icon="⚽",
    layout="wide"
)

@st.cache_data
def load_data():
    
    data_path = "E0.csv"
    
    # Try to find the data file
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
    
    # Add some calculated columns
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
    
    st.title("⚽ Soccer League Data Overview")
    
    df = load_data()
    if df is None:
        st.error("Unable to load data.")
        return
    
    # Basic dataset info
    st.header("📊 Dataset Summary")
    
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
        st.metric("Total Goals", df['Total_Goals'].sum())
    
    # League overview stats
    st.header("🏆 League Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Match results pie chart
        result_counts = df['Result'].value_counts()
        fig_results = px.pie(values=result_counts.values, names=result_counts.index,
                            title="Match Results Distribution")
        st.plotly_chart(fig_results, use_container_width=True)
    
    with col2:
        # Goals per match histogram
        fig_goals = px.histogram(df, x='Total_Goals', nbins=10,
                               title="Goals per Match Distribution")
        st.plotly_chart(fig_goals, use_container_width=True)
    
    # Team performance charts
    st.header("Top Performers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most wins
        home_wins = df[df['Result'] == 'Home Win']['HomeTeam'].value_counts()
        away_wins = df[df['Result'] == 'Away Win']['AwayTeam'].value_counts()
        total_wins = (home_wins.fillna(0) + away_wins.fillna(0)).sort_values(ascending=False).head(10)
        
        fig_wins = px.bar(x=total_wins.index, y=total_wins.values,
                         title="Teams with Most Wins")
        fig_wins.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_wins, use_container_width=True)
    
    with col2:
        # Highest scoring teams
        home_goals = df.groupby('HomeTeam')['FTHG'].sum()
        away_goals = df.groupby('AwayTeam')['FTAG'].sum()
        total_goals = (home_goals.fillna(0) + away_goals.fillna(0)).sort_values(ascending=False).head(10)
        
        fig_goals_team = px.bar(x=total_goals.index, y=total_goals.values,
                               title="Highest Scoring Teams")
        fig_goals_team.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_goals_team, use_container_width=True)
    
    # Show recent matches
    st.header("Recent Matches Sample")
    
    display_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result']
    available_cols = [col for col in display_cols if col in df.columns]
    
    if 'Date' in df.columns:
        recent_matches = df.sort_values('Date', ascending=False).head(10)
    else:
        recent_matches = df.head(10)
    
    st.dataframe(recent_matches[available_cols], use_container_width=True)

def page_team_analysis():
    
    st.title("Team Analysis")
    
    df = load_data()
    if df is None:
        return
    
    # Team selection
    all_teams = sorted(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))
    
    selected_team = st.selectbox("Select Team for Analysis:", all_teams)
    
    if not selected_team:
        st.warning("Please select a team to view analytics.")
        return
    
    # Get team data
    team_data = get_team_data(df, selected_team)
    
    if len(team_data) == 0:
        st.error(f"No matches found for {selected_team}")
        return
    
    st.header(f"📊 {selected_team} Performance Overview")
    
    # Basic team stats
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_matches = len(team_data)
    wins = len(team_data[team_data['Team_Result'] == 'Win'])
    draws = len(team_data[team_data['Team_Result'] == 'Draw'])
    losses = len(team_data[team_data['Team_Result'] == 'Loss'])
    goals_scored = team_data['Team_Goals'].sum()
    
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
    
    # More detailed stats
    col1, col2, col3, col4 = st.columns(4)
    
    goals_conceded = team_data['Opponent_Goals'].sum()
    
    with col1:
        st.metric("Goals Scored", goals_scored)
    with col2:
        st.metric("Goals Conceded", goals_conceded)
    with col3:
        goal_diff = goals_scored - goals_conceded
        st.metric("Goal Difference", f"{int(goal_diff):+.0f}")
    with col4:
        avg_goals = goals_scored / total_matches if total_matches > 0 else 0
        st.metric("Goals per Match", f"{avg_goals:.1f}")
    
    # Performance charts
    st.header("📈 Performance Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Win/Loss/Draw pie chart
        result_counts = team_data['Team_Result'].value_counts()
        fig_results = px.pie(values=result_counts.values, names=result_counts.index,
                           title=f"{selected_team} - Match Results")
        st.plotly_chart(fig_results, use_container_width=True)
    
    with col2:
        # Home vs Away performance
        venue_performance = team_data.groupby(['Venue', 'Team_Result']).size().unstack(fill_value=0)
        fig_venue = px.bar(venue_performance, 
                          title=f"{selected_team} - Home vs Away Performance")
        st.plotly_chart(fig_venue, use_container_width=True)
    
    # Goals analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Goals scored distribution
        fig_goals_scored = px.histogram(team_data, x='Team_Goals', nbins=8,
                                       title=f"{selected_team} - Goals Scored per Match")
        st.plotly_chart(fig_goals_scored, use_container_width=True)
    
    with col2:
        # Goals scored vs conceded scatter plot
        fig_scatter = px.scatter(team_data, x='Team_Goals', y='Opponent_Goals',
                               color='Team_Result',
                               title=f"{selected_team} - Goals For vs Against")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Performance over time (if date available)
    if 'Date' in team_data.columns and not team_data['Date'].isna().all():
        st.header("📅 Performance Over Time")
        
        # Sort by date and create cumulative stats
        team_data_sorted = team_data.sort_values('Date').reset_index(drop=True)
        team_data_sorted['Points'] = team_data_sorted['Team_Result'].map(
            {'Win': 3, 'Draw': 1, 'Loss': 0})
        team_data_sorted['Cumulative_Points'] = team_data_sorted['Points'].cumsum()
        team_data_sorted['Match_Number'] = range(1, len(team_data_sorted) + 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Points accumulation
            fig_points = px.line(team_data_sorted, x='Match_Number', y='Cumulative_Points',
                               title=f"{selected_team} - Points Accumulation",
                               markers=True)
            st.plotly_chart(fig_points, use_container_width=True)
        
        with col2:
            # Rolling average goals
            team_data_sorted['Rolling_Avg_Goals'] = team_data_sorted['Team_Goals'].rolling(window=5, min_periods=1).mean()
            fig_rolling = px.line(team_data_sorted, x='Match_Number', y='Rolling_Avg_Goals',
                                title=f"{selected_team} - Rolling Average Goals (5 matches)",
                                markers=True)
            st.plotly_chart(fig_rolling, use_container_width=True)
    
    # Recent form
    st.header("Recent Form")
    
    # Get last 10 matches
    if 'Date' in team_data.columns and not team_data['Date'].isna().all():
        recent_matches = team_data.sort_values('Date', ascending=False).head(10)
    else:
        recent_matches = team_data.tail(10)
    
    # Show form as colored circles
    recent_results = recent_matches['Team_Result'].tolist()
    form_colors = {'Win': '🟢', 'Draw': '🟡', 'Loss': '🔴'}
    form_string = ' '.join([form_colors[result] for result in reversed(recent_results)])
    
    st.write(f"**Last {len(recent_results)} matches:** {form_string}")
    
    # Recent matches table
    display_cols = ['Date', 'Opponent', 'Venue', 'Team_Goals', 'Opponent_Goals', 'Team_Result']
    available_cols = [col for col in display_cols if col in recent_matches.columns]
    
    st.dataframe(recent_matches[available_cols].reset_index(drop=True), use_container_width=True)

def page_team_comparison():
    
    st.title("🆚 Team Comparison")
    
    df = load_data()
    if df is None:
        return
    
    all_teams = sorted(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("Select First Team:", all_teams, key="team1")
    
    with col2:
        team2 = st.selectbox("Select Second Team:", all_teams, key="team2")
    
    if not team1 or not team2 or team1 == team2:
        st.warning("Select two different teams for comparison.")
        return
    
    # Get data for both teams
    team1_data = get_team_data(df, team1)
    team2_data = get_team_data(df, team2)
    
    st.header(f"📊 {team1} vs {team2} Comparison")
    
    # Calculate comparison metrics
    def calculate_team_stats(team_data):
        total_matches = len(team_data)
        wins = len(team_data[team_data['Team_Result'] == 'Win'])
        draws = len(team_data[team_data['Team_Result'] == 'Draw'])
        losses = len(team_data[team_data['Team_Result'] == 'Loss'])
        
        return {
            'Matches': total_matches,
            'Wins': wins,
            'Draws': draws,
            'Losses': losses,
            'Win Rate (%)': (wins / total_matches * 100) if total_matches > 0 else 0,
            'Goals Scored': team_data['Team_Goals'].sum(),
            'Goals Conceded': team_data['Opponent_Goals'].sum(),
            'Goals/Match': team_data['Team_Goals'].mean(),
            'Conceded/Match': team_data['Opponent_Goals'].mean(),
            'Points': wins * 3 + draws,
            'Points/Match': (wins * 3 + draws) / total_matches if total_matches > 0 else 0
        }
    
    team1_stats = calculate_team_stats(team1_data)
    team2_stats = calculate_team_stats(team2_data)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Metric': list(team1_stats.keys()),
        team1: list(team1_stats.values()),
        team2: list(team2_stats.values())
    })
    
    # Round numerical values
    for col in [team1, team2]:
        comparison_df[col] = comparison_df[col].round(2)
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Comparison charts
    st.header("📈 Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Win rate comparison
        win_rates = [team1_stats['Win Rate (%)'], team2_stats['Win Rate (%)']]
        fig_win_rate = px.bar(x=[team1, team2], y=win_rates,
                             title="Win Rate Comparison (%)")
        st.plotly_chart(fig_win_rate, use_container_width=True)
    
    with col2:
        # Goals per match comparison
        goals_per_match = [team1_stats['Goals/Match'], team2_stats['Goals/Match']]
        fig_goals = px.bar(x=[team1, team2], y=goals_per_match,
                          title="Goals per Match Comparison")
        st.plotly_chart(fig_goals, use_container_width=True)
    
    # Head-to-head record
    st.header("Head-to-Head Record")
    
    h2h_matches = df[((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) |
                    ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))]
    
    if len(h2h_matches) > 0:
        # Calculate H2H stats
        team1_wins = len(h2h_matches[((h2h_matches['HomeTeam'] == team1) & (h2h_matches['FTHG'] > h2h_matches['FTAG'])) |
                                   ((h2h_matches['AwayTeam'] == team1) & (h2h_matches['FTAG'] > h2h_matches['FTHG']))])
        team2_wins = len(h2h_matches[((h2h_matches['HomeTeam'] == team2) & (h2h_matches['FTHG'] > h2h_matches['FTAG'])) |
                                   ((h2h_matches['AwayTeam'] == team2) & (h2h_matches['FTAG'] > h2h_matches['FTHG']))])
        draws = len(h2h_matches[h2h_matches['FTHG'] == h2h_matches['FTAG']])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{team1} Wins", team1_wins)
        with col2:
            st.metric("Draws", draws)
        with col3:
            st.metric(f"{team2} Wins", team2_wins)
        
        # Recent H2H matches
        st.subheader("Recent Head-to-Head Matches")
        h2h_display = h2h_matches[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result']].tail(5)
        st.dataframe(h2h_display, use_container_width=True)
        
    else:
        st.info("No head-to-head matches found between these teams.")

def page_prediction():
    
    st.title("Match Prediction")
    
    df = load_data()
    if df is None:
        return
    
    # Simple model training
    @st.cache_resource
    def train_prediction_model(df):
        
        df_ml = df.copy()
        
        # Encode teams
        le_home = LabelEncoder()
        le_away = LabelEncoder()
        
        df_ml['HomeTeam_Encoded'] = le_home.fit_transform(df_ml['HomeTeam'])
        df_ml['AwayTeam_Encoded'] = le_away.fit_transform(df_ml['AwayTeam'])
        
        # Features and target
        X = df_ml[['HomeTeam_Encoded', 'AwayTeam_Encoded']]
        y = df_ml['Result']
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, le_home, le_away, accuracy
    
    # Train the model
    with st.spinner("Training prediction model"):
        model, le_home, le_away, accuracy = train_prediction_model(df)
    
    st.success(f"Model Accuracy: {accuracy:.1%}")
    
    st.header("Predict Match Outcome")
    
    # Team selection
    all_teams = sorted(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Home Team")
        home_team = st.selectbox("Select Home Team:", all_teams, key="home")
    
    with col2:
        st.subheader("Away Team")
        away_team = st.selectbox("Select Away Team:", all_teams, key="away")
    
    # Prediction
    if st.button("🎯 Predict Match Outcome", type="primary"):
        if home_team == away_team:
            st.error("Please select different teams!")
        else:
            # Encode teams
            home_encoded = le_home.transform([home_team])[0]
            away_encoded = le_away.transform([away_team])[0]
            
            # Make prediction
            prediction_input = np.array([[home_encoded, away_encoded]])
            prediction = model.predict(prediction_input)[0]
            probabilities = model.predict_proba(prediction_input)[0]
            
            # Display results
            st.header("Prediction Results")
            
            # Main prediction
            st.subheader(f"{home_team} vs {away_team}")
            st.write(f"**Predicted Outcome: {prediction}**")
            st.write(f"**Confidence: {max(probabilities):.1%}**")
            
            # Probability breakdown
            st.subheader("📊 Outcome Probabilities")
            
            classes = model.classes_
            prob_dict = dict(zip(classes, probabilities))
            
            for outcome, prob in prob_dict.items():
                st.write(f"**{outcome}:** {prob:.1%}")
                st.progress(prob)
            
            # Probability chart
            fig_probs = px.bar(x=list(prob_dict.keys()), y=list(prob_dict.values()),
                              title="Match Outcome Probabilities")
            st.plotly_chart(fig_probs, use_container_width=True)
            
            # Show team stats for context
            st.header("📊 Team Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{home_team} Stats")
                home_data = get_team_data(df, home_team)
                home_home_data = home_data[home_data['Venue'] == 'Home']
                
                if len(home_home_data) > 0:
                    home_win_rate = len(home_home_data[home_home_data['Team_Result'] == 'Win']) / len(home_home_data) * 100
                    home_goals_avg = home_home_data['Team_Goals'].mean()
                    
                    st.write(f"Home Win Rate: {home_win_rate:.1f}%")
                    st.write(f"Home Goals/Game: {home_goals_avg:.1f}")
            
            with col2:
                st.subheader(f"{away_team} Stats")
                away_data = get_team_data(df, away_team)
                away_away_data = away_data[away_data['Venue'] == 'Away']
                
                if len(away_away_data) > 0:
                    away_win_rate = len(away_away_data[away_away_data['Team_Result'] == 'Win']) / len(away_away_data) * 100
                    away_goals_avg = away_away_data['Team_Goals'].mean()
                    
                    st.write(f"Away Win Rate: {away_win_rate:.1f}%")
                    st.write(f"Away Goals/Game: {away_goals_avg:.1f}")

def main():
    
    
    # Simple sidebar navigation
    st.sidebar.title("⚽ Soccer Analytics")
    
    pages = {
        "Data Overview": page_data_overview,
        "Team Analysis": page_team_analysis,
        "Team Comparison": page_team_comparison,
        "Match Prediction": page_prediction
    }
    
    selected_page = st.sidebar.selectbox("Pages:", list(pages.keys()))
    
    # Run the selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()