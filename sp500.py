import requests
from bs4 import BeautifulSoup
import json


def get_sp500_symbols():
    sp500_wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(sp500_wiki_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        symbols = [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:]]
        return symbols
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return []

def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)

if __name__ == "__main__":
    sp500_symbols = get_sp500_symbols()
    print("S&P 500 Stock Symbols:")
    print(sp500_symbols)

    json_filename = 'sp500_symbols.json'
    save_to_json(sp500_symbols, json_filename)
    print(f"Symbols saved to {json_filename}")

