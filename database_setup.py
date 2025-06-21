
import pandas as pd
import sqlite3
import os
from datetime import datetime

def create_database(csv_path="Club-Football-Match-Data-2000-2025/data/E0.csv", 
                   db_path="soccer_matches.db"):
    
    try:
        
        print("Loading data from CSV...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        
        print("Preprocessing data...")
        
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        
        df['Total_Goals'] = df['FTHG'] + df['FTAG']
        df['Goal_Difference'] = df['FTHG'] - df['FTAG']
        
        
        def get_result(row):
            if row['FTHG'] > row['FTAG']:
                return 'Home Win'
            elif row['FTHG'] < row['FTAG']:
                return 'Away Win'
            else:
                return 'Draw'
        
        df['Result'] = df.apply(get_result, axis=1)
        df['High_Scoring'] = (df['Total_Goals'] > 2.5).astype(int)
        
        
        print("Creating database...")
        conn = sqlite3.connect(db_path)
        
        
        df.to_sql('matches', conn, if_exists='replace', index=False)
        print("Main matches table created")
        
     
        create_team_stats_table(df, conn)
        
      
        create_indexes(conn)
        
        conn.close()
        print(f"Database created successfully at: {db_path}")
        return True
        
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

def create_team_stats_table(df, conn):
    
    print("Creating team statistics table...")
    
  
    home_stats = df.groupby('HomeTeam').agg({
        'FTHG': ['count', 'sum', 'mean'],
        'FTAG': 'sum',
        'Result': lambda x: (x == 'Home Win').sum()
    }).round(2)
    
    home_stats.columns = ['Games_Played', 'Goals_For', 'Avg_Goals_For', 
                         'Goals_Against', 'Wins']
    home_stats = home_stats.reset_index()
    home_stats['Team'] = home_stats['HomeTeam']
    home_stats['Venue'] = 'Home'
    
  
    away_stats = df.groupby('AwayTeam').agg({
        'FTAG': ['count', 'sum', 'mean'],
        'FTHG': 'sum',
        'Result': lambda x: (x == 'Away Win').sum()
    }).round(2)
    
    away_stats.columns = ['Games_Played', 'Goals_For', 'Avg_Goals_For', 
                         'Goals_Against', 'Wins']
    away_stats = away_stats.reset_index()
    away_stats['Team'] = away_stats['AwayTeam']
    away_stats['Venue'] = 'Away'
    
    
    team_stats = pd.concat([
        home_stats[['Team', 'Venue', 'Games_Played', 'Goals_For', 
                   'Avg_Goals_For', 'Goals_Against', 'Wins']],
        away_stats[['Team', 'Venue', 'Games_Played', 'Goals_For', 
                   'Avg_Goals_For', 'Goals_Against', 'Wins']]
    ])
    
    team_stats.to_sql('team_statistics', conn, if_exists='replace', index=False)
    print("Team statistics table created")

def create_indexes(conn):
    
    print("Creating database indexes...")
    
    cursor = conn.cursor()
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_home_team ON matches(HomeTeam)",
        "CREATE INDEX IF NOT EXISTS idx_away_team ON matches(AwayTeam)",
        "CREATE INDEX IF NOT EXISTS idx_date ON matches(Date)",
        "CREATE INDEX IF NOT EXISTS idx_result ON matches(Result)",
        "CREATE INDEX IF NOT EXISTS idx_total_goals ON matches(Total_Goals)"
    ]
    
    for index in indexes:
        cursor.execute(index)
    
    conn.commit()
    print("Database indexes created")

def main():
    
    print("Soccer Prediction App - Database Setup")
    print("=" * 50)
    
    csv_path = "Club-Football-Match-Data-2000-2025/data/E0.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at {csv_path}")
        print("Please run data_downloader.py first to get the data.")
        return
    
    success = create_database(csv_path)
    
    if success:
        print("\n Database setup completed successfully!")
    else:
        print("Database setup failed.")

if __name__ == "__main__":
    main()