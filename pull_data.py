import urllib.request
import json
import os
import requests
from bs4 import BeautifulSoup
import time

BASE_URL = "https://www.cross-tables.com"
LIST_URL = "https://www.cross-tables.com/annolistself.php?offset="
DOWNLOAD_DIR = "dataset/gcg_files"
GAME_IDS_FILE = "dataset/game_ids.json"

def get_game_ids():
    os.makedirs("dataset", exist_ok=True)
    if os.path.exists(GAME_IDS_FILE):
        with open(GAME_IDS_FILE, 'r') as f:
            ids = json.load(f)
            print(f"Loaded {len(ids)} game IDs from {GAME_IDS_FILE}")
            return list(ids)

    game_ids = set()
    offset = 1
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    }
    
    print("Beginning to scrape game IDs from annolistself.php...")
    try:
        while True:
            url = f"{LIST_URL}{offset}"
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to fetch {url} - Status Code {response.status_code}")
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            
            found_on_page = 0
            for link in links:
                href = link['href']
                if 'annotated.php?u=' in href:
                    game_id = href.split('u=')[1].split('&')[0]
                    if game_id.isdigit():
                        if game_id not in game_ids:
                            game_ids.add(game_id)
                            found_on_page += 1
                            
            print(f"Offset {offset}: found {found_on_page} new games (Total: {len(game_ids)})")
            
            if found_on_page == 0:
                break
                
            offset += 100
            time.sleep(1) # Be respectful
    except KeyboardInterrupt:
        print("Interrupted by user. Saving progress...")
        
    game_ids_list = list(game_ids)
    with open(GAME_IDS_FILE, 'w') as f:
        json.dump(game_ids_list, f)
    
    return game_ids_list

from concurrent.futures import ThreadPoolExecutor

def download_gcg_files(game_ids):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    print(f"\nStarting download of {len(game_ids)} .gcg files...")
    
    # Use connection pooling
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    })
    
    def fetch_file(idx_game_id):
        idx, game_id = idx_game_id
        prefix = int(game_id) // 100
        gcg_url = f"{BASE_URL}/annotated/selfgcg/{prefix}/anno{game_id}.gcg"
        
        target_file = os.path.join(DOWNLOAD_DIR, f"anno{game_id}.gcg")
        if os.path.exists(target_file):
            return
            
        try:
            response = session.get(gcg_url, timeout=10)
            if response.status_code == 200:
                with open(target_file, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {gcg_url} - Status {response.status_code}")
        except Exception as e:
            print(f"Error downloading {gcg_url}: {e}")
            
        if (idx + 1) % 500 == 0:
            print(f"Downloaded {idx + 1}/{len(game_ids)}...")

    # Use 10 threads to be respectful but fast
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetch_file, enumerate(game_ids))

if __name__ == "__main__":
    ids = get_game_ids()
    print(f"\nTotal unique game IDs found: {len(ids)}")
    if ids:
        download_gcg_files(ids)
        print("Data pulling complete!")
