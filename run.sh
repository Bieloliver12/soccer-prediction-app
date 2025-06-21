#!/bin/bash
echo "🚀 Starting Soccer Prediction App..."

# Activate virtual environment
source soccer_env/bin/activate

# Check if data exists, download if needed
if [ ! -f "Club-Football-Match-Data-2000-2025/data/E0.csv" ]; then
    echo "📥 Downloading soccer data..."
    python data_downloader.py
fi

# Check if database exists, create if needed
if [ ! -f "soccer_matches.db" ]; then
    echo "🗄️ Creating database..."
    python database_setup.py
fi

# Run the app
echo "🌐 Launching Streamlit app..."
streamlit run app.py --server.port 8501

echo "App available at http://localhost:8501"