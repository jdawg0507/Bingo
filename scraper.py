import requests
from bs4 import BeautifulSoup
import time

def fetch_page():
    url = "https://www.cross-tables.com/annotated.php?u=59483"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        with open("debug_annotated.html", "w") as f:
            f.write(response.text)
        print("Written response to debug_annotated.html")

if __name__ == "__main__":
    fetch_page()
