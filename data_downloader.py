
import os
import requests
import pandas as pd
from pathlib import Path
import time
import json

class SoccerDataDownloader:
    def __init__(self):
        self.data_dir = Path("Club-Football-Match-Data-2000-2025/data")
        
        
        self.data_sources = {
            "football_data_uk": {
                "name": "Football-Data.co.uk (Official)",
                "base_url": "https://www.football-data.co.uk/mmz4281/",
                "seasons": [
                    "0001", "0102", "0203", "0304", "0405", "0506", "0607", "0708", "0809", "0910",
                    "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819", "1920", 
                    "2021", "2122", "2223", "2324", "2425"
                ],
                "league_code": "E0"
            },
            "github_football_csv": {
                "name": "Football CSV (GitHub)",
                "urls": [
                    "https://raw.githubusercontent.com/footballcsv/england/master/2020s/2023-24/eng.1.csv",
                    "https://raw.githubusercontent.com/footballcsv/england/master/2020s/2022-23/eng.1.csv", 
                    "https://raw.githubusercontent.com/footballcsv/england/master/2020s/2021-22/eng.1.csv",
                    "https://raw.githubusercontent.com/footballcsv/england/master/2010s/2019-20/eng.1.csv",
                    "https://raw.githubusercontent.com/footballcsv/england/master/2010s/2018-19/eng.1.csv",
                    "https://raw.githubusercontent.com/footballcsv/england/master/2010s/2017-18/eng.1.csv",
                    "https://raw.githubusercontent.com/footballcsv/england/master/2010s/2016-17/eng.1.csv",
                    "https://raw.githubusercontent.com/footballcsv/england/master/2010s/2015-16/eng.1.csv"
                ]
            },
            "kaggle_datasets": {
                "name": "Kaggle Soccer Datasets",
                "urls": [
                    "https://raw.githubusercontent.com/datasets/football-results/master/data/english-premier-league-2020-2021.csv",
                    "https://raw.githubusercontent.com/datasets/football-results/master/data/english-premier-league-2019-2020.csv",
                    "https://raw.githubusercontent.com/datasets/football-results/master/data/english-premier-league-2018-2019.csv"
                ]
            },
            "openfootball": {
                "name": "Open Football Data",
                "urls": [
                    "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/en.1.json",
                    "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/en.1.json",
                    "https://raw.githubusercontent.com/openfootball/football.json/master/2021-22/en.1.json"
                ]
            }
        }
        
    def create_directories(self):
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {self.data_dir}")
    
    def standardize_column_names(self, df, source_type="football_data_uk"):
    
        
        
        column_mappings = {
            "football_data_uk": {
                
            },
            "github_football_csv": {
                "Date": "Date",
                "Team 1": "HomeTeam", 
                "Team 2": "AwayTeam",
                "FT": "FTR",
                "Score": "Score"
            },
            "kaggle_datasets": {
                "home_team": "HomeTeam",
                "away_team": "AwayTeam", 
                "home_score": "FTHG",
                "away_score": "FTAG",
                "date": "Date"
            },
            "openfootball": {
                
            }
        }
        
        if source_type in column_mappings:
            df = df.rename(columns=column_mappings[source_type])
        
        
        if 'Score' in df.columns and 'FTHG' not in df.columns:
            try:
                score_split = df['Score'].str.split('-', expand=True)
                df['FTHG'] = pd.to_numeric(score_split[0], errors='coerce')
                df['FTAG'] = pd.to_numeric(score_split[1], errors='coerce')
            except:
                pass
        
        
        required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f" Missing required columns: {missing_cols}")
            return None
        
        return df
    
    def download_from_football_data_uk(self):
        
        print(" Trying Football-Data.co.uk (Official Premier League Data)...")
        
        source = self.data_sources["football_data_uk"]
        downloaded_files = []
        
        for season in source["seasons"]:
            url = f"{source['base_url']}{season}/{source['league_code']}.csv"
            file_path = self.data_dir / f"E0_{season}.csv"
            
            try:
                print(f" Downloading season 20{season[:2]}-{season[2:]}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                
                test_df = pd.read_csv(file_path)
                if len(test_df) > 50:  
                    downloaded_files.append(file_path)
                    print(f"✅ Season 20{season[:2]}-{season[2:]}: {len(test_df)} matches")
                else:
                    print(f" Season 20{season[:2]}-{season[2:]}: Insufficient data ({len(test_df)} matches)")
                    
            except Exception as e:
                print(f"❌ Season 20{season[:2]}-{season[2:]}: {e}")
                
            time.sleep(0.5)  
        
        return downloaded_files
    
    def download_from_github_football_csv(self):
        
        print(" Trying GitHub Football CSV repository...")
        
        source = self.data_sources["github_football_csv"]
        downloaded_files = []
        
        for i, url in enumerate(source["urls"]):
            try:
                print(f" Downloading from GitHub CSV source {i+1}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                
                df = pd.read_csv(url)
                df = self.standardize_column_names(df, "github_football_csv")
                
                if df is not None and len(df) > 20:
                    file_path = self.data_dir / f"github_csv_{i+1}.csv"
                    df.to_csv(file_path, index=False)
                    downloaded_files.append(file_path)
                    print(f"✅ GitHub CSV {i+1}: {len(df)} matches")
                else:
                    print(f" GitHub CSV {i+1}: Invalid data format")
                    
            except Exception as e:
                print(f"❌ GitHub CSV {i+1}: {e}")
                
            time.sleep(0.5)
        
        return downloaded_files
    
    def download_from_kaggle_datasets(self):
    
        print(" Trying Kaggle/GitHub datasets...")
        
        source = self.data_sources["kaggle_datasets"]
        downloaded_files = []
        
        for i, url in enumerate(source["urls"]):
            try:
                print(f" Downloading Kaggle dataset {i+1}...")
                
                df = pd.read_csv(url)
                df = self.standardize_column_names(df, "kaggle_datasets")
                
                if df is not None and len(df) > 20:
                    file_path = self.data_dir / f"kaggle_{i+1}.csv"
                    df.to_csv(file_path, index=False)
                    downloaded_files.append(file_path)
                    print(f"✅ Kaggle dataset {i+1}: {len(df)} matches")
                else:
                    print(f" Kaggle dataset {i+1}: Invalid data format")
                    
            except Exception as e:
                print(f"❌ Kaggle dataset {i+1}: {e}")
                
            time.sleep(0.5)
        
        return downloaded_files
    
    def download_from_openfootball(self):
        
        print(" Trying Open Football JSON data...")
        
        source = self.data_sources["openfootball"]
        downloaded_files = []
        
        for i, url in enumerate(source["urls"]):
            try:
                print(f" Downloading Open Football JSON {i+1}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Parse JSON
                json_data = response.json()
                matches = []
                
                # Extract matches from JSON structure
                if 'rounds' in json_data:
                    for round_data in json_data['rounds']:
                        if 'matches' in round_data:
                            for match in round_data['matches']:
                                if 'team1' in match and 'team2' in match and 'score' in match:
                                    score = match['score']
                                    if 'ft' in score and len(score['ft']) == 2:
                                        matches.append({
                                            'Date': match.get('date', ''),
                                            'HomeTeam': match['team1'],
                                            'AwayTeam': match['team2'],
                                            'FTHG': score['ft'][0],
                                            'FTAG': score['ft'][1]
                                        })
                
                if len(matches) > 20:
                    df = pd.DataFrame(matches)
                    file_path = self.data_dir / f"openfootball_{i+1}.csv"
                    df.to_csv(file_path, index=False)
                    downloaded_files.append(file_path)
                    print(f"✅ Open Football JSON {i+1}: {len(matches)} matches")
                else:
                    print(f" Open Football JSON {i+1}: Insufficient matches ({len(matches)})")
                    
            except Exception as e:
                print(f"❌ Open Football JSON {i+1}: {e}")
                
            time.sleep(0.5)
        
        return downloaded_files
    
    def try_alternative_sources(self):
        
        
        
        alternative_urls = [
            
            "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv",
            
            
            "https://raw.githubusercontent.com/hugomathien/soccer-predictions/master/data/sample_data.csv",
            
            
            "https://raw.githubusercontent.com/jalapic/engsoccerdata/master/data-raw/england_current.csv",
            
            
            "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
        ]
        
        downloaded_files = []
        
        for i, url in enumerate(alternative_urls):
            try:
                print(f" Trying alternative source {i+1}...")
                
                df = pd.read_csv(url)
                
                
                possible_home_cols = ['home_team', 'HomeTeam', 'Home', 'team1', 'home']
                possible_away_cols = ['away_team', 'AwayTeam', 'Away', 'team2', 'away']
                possible_home_score = ['home_score', 'FTHG', 'score1', 'home_goals']
                possible_away_score = ['away_score', 'FTAG', 'score2', 'away_goals']
                
                home_col = next((col for col in possible_home_cols if col in df.columns), None)
                away_col = next((col for col in possible_away_cols if col in df.columns), None)
                home_score_col = next((col for col in possible_home_score if col in df.columns), None)
                away_score_col = next((col for col in possible_away_score if col in df.columns), None)
                
                if all([home_col, away_col, home_score_col, away_score_col]):
                    df_clean = df[[home_col, away_col, home_score_col, away_score_col]].copy()
                    df_clean.columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
                    
                   
                    date_cols = ['date', 'Date', 'match_date', 'game_date']
                    date_col = next((col for col in date_cols if col in df.columns), None)
                    if date_col:
                        df_clean['Date'] = df[date_col]
                    
                    if len(df_clean) > 100:
                        file_path = self.data_dir / f"alternative_{i+1}.csv"
                        df_clean.to_csv(file_path, index=False)
                        downloaded_files.append(file_path)
                        print(f"✅ Alternative source {i+1}: {len(df_clean)} matches")
                    else:
                        print(f" Alternative source {i+1}: Insufficient data")
                else:
                    print(f" Alternative source {i+1}: Cannot identify required columns")
                    
            except Exception as e:
                print(f"❌ Alternative source {i+1}: {e}")
                
            time.sleep(1)
        
        return downloaded_files
    
    def combine_all_sources(self):
    
        print("\n Combining data from all sources...")
        
        all_files = list(self.data_dir.glob("*.csv"))
        if not all_files:
            print("❌ No data files found to combine")
            return None
        
        combined_data = []
        total_matches = 0
        sources_used = []
        
        for file_path in all_files:
            if file_path.name == "E0.csv":  
                continue
                
            try:
                df = pd.read_csv(file_path)
                
                
                required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
                if all(col in df.columns for col in required_cols):
                    
                    df['Source'] = file_path.stem
                    
                    
                    df = df.dropna(subset=required_cols)
                    df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
                    df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
                    df = df.dropna(subset=['FTHG', 'FTAG'])
                    
                    if len(df) > 0:
                        combined_data.append(df)
                        total_matches += len(df)
                        sources_used.append(file_path.stem)
                        print(f"✅ {file_path.name}: {len(df)} matches")
                else:
                    print(f" {file_path.name}: Missing required columns")
                    
            except Exception as e:
                print(f"❌ {file_path.name}: {e}")
        
        if not combined_data:
            print("❌ No valid data found to combine")
            return None
        
        
        master_df = pd.concat(combined_data, ignore_index=True)
        
        
        duplicate_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        if 'Date' in master_df.columns:
            duplicate_cols.append('Date')
        
        initial_count = len(master_df)
        master_df = master_df.drop_duplicates(subset=duplicate_cols)
        final_count = len(master_df)
        
        print(f" Removed {initial_count - final_count} duplicate matches")
        
        
        master_file = self.data_dir / "E0.csv"
        master_df.to_csv(master_file, index=False)
        
        print(f"\n REAL DATA COMBINED SUCCESSFULLY!")
        print(f"📊 Total matches: {len(master_df):,}")
        print(f" Unique teams: {len(set(master_df['HomeTeam'].unique().tolist() + master_df['AwayTeam'].unique().tolist()))}")
        print(f" Variables: {len(master_df.columns)}")
        print(f" Sources used: {len(sources_used)} ({', '.join(sources_used[:5])}{'...' if len(sources_used) > 5 else ''})")
        print(f" Saved to: {master_file}")
        
        return master_file
    
    def download_real_data(self):
        
        print("⚽ REAL SOCCER DATA DOWNLOADER")
        print("=" * 50)
        print("=" * 50)
        
        self.create_directories()
        
        all_downloaded_files = []
        
        
        try:
            files = self.download_from_football_data_uk()
            all_downloaded_files.extend(files)
        except Exception as e:
            print(f"❌ Football-Data.co.uk failed: {e}")
        
        try:
            files = self.download_from_github_football_csv()
            all_downloaded_files.extend(files)
        except Exception as e:
            print(f"❌ GitHub Football CSV failed: {e}")
        
        try:
            files = self.download_from_kaggle_datasets()
            all_downloaded_files.extend(files)
        except Exception as e:
            print(f"❌ Kaggle datasets failed: {e}")
        
        try:
            files = self.download_from_openfootball()
            all_downloaded_files.extend(files)
        except Exception as e:
            print(f"❌ Open Football failed: {e}")
        
        try:
            files = self.try_alternative_sources()
            all_downloaded_files.extend(files)
        except Exception as e:
            print(f"❌ Alternative sources failed: {e}")
        
        if not all_downloaded_files:
            print("❌ No data sources were successful")
            return None
        
        print(f"\n✅ Successfully downloaded from {len(all_downloaded_files)} sources")
        
        
        master_file = self.combine_all_sources()
        return master_file

def main():
    
    downloader = SoccerDataDownloader()
    
    
    print("• Football-Data.co.uk (Official Premier League data)")
    print("• GitHub Football CSV repositories")
    print("• Kaggle soccer datasets")
    print("• Open Football JSON data")
    print("• FiveThirtyEight soccer data")
    print("• Various GitHub soccer data repositories")
   
    
    
    result = downloader.download_real_data()
    
    if result:
        print(f"\n SUCCESS!")
        
        
        df = pd.read_csv(result)
        print(f"\n📊 FINAL DATASET:")
        print(f"Matches: {len(df):,}")
        print(f"Teams: {len(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))}")
        print(f"Variables: {len(df.columns)}")
        if 'Date' in df.columns:
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        if 'Source' in df.columns:
            print(f"Sources: {df['Source'].nunique()} different data sources")
        
        print("\n✅ Run: streamlit run app.py")
    else:
        print("\n❌ Failed to download real soccer data")
        print("Check your internet connection and try again")

if __name__ == "__main__":
    main()