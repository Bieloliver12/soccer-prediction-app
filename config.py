"""
Configuration file for Soccer Prediction App
Contains all configuration parameters and constants
"""

import os

# File paths
DATA_PATH = "Club-Football-Match-Data-2000-2025/data/E0.csv"
DATABASE_PATH = "soccer_matches.db"
MODEL_PATH = "soccer_model.pkl"
HOME_ENCODER_PATH = "le_home.pkl"
AWAY_ENCODER_PATH = "le_away.pkl"

# Database configuration
DB_CONFIG = {
    'database': DATABASE_PATH,
    'timeout': 30,
    'check_same_thread': False
}

# Model configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 100,
    'max_iter': 1000
}

# Feature columns configuration
BASIC_FEATURES = ['HomeTeam_Encoded', 'AwayTeam_Encoded']

OPTIONAL_FEATURES = [
    'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR'
]

# UI configuration
UI_CONFIG = {
    'page_title': "Soccer Match Predictor",
    'page_icon': "⚽",
    'layout': "wide",
    'sidebar_state': "expanded"
}

# SQL Queries
SAMPLE_QUERIES = {
    "All Matches": "SELECT * FROM matches LIMIT 10",
    "Home Wins": """
        SELECT HomeTeam, COUNT(*) as Wins 
        FROM matches 
        WHERE FTHG > FTAG 
        GROUP BY HomeTeam 
        ORDER BY Wins DESC
    """,
    "High Scoring Games": """
        SELECT HomeTeam, AwayTeam, FTHG, FTAG, Total_Goals
        FROM matches 
        WHERE Total_Goals > 4
        ORDER BY Total_Goals DESC
    """,
    "Team Statistics": """
        SELECT HomeTeam as Team, 
               COUNT(*) as Home_Games,
               AVG(FTHG) as Avg_Home_Goals,
               SUM(CASE WHEN FTHG > FTAG THEN 1 ELSE 0 END) as Home_Wins
        FROM matches 
        GROUP BY HomeTeam
        ORDER BY Home_Wins DESC
    """
}

def get_feature_columns(df):
    #Get available feature columns from dataframe
    available_features = BASIC_FEATURES.copy()
    
    for feature in OPTIONAL_FEATURES:
        if feature in df.columns:
            available_features.append(feature)
    
    return available_features