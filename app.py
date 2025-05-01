# This script fetches data from the Market Index API and saves the filtered data to a JSON file.
import json
import requests


# URL for the API request
URL = "https://quoteapi.com/api/v5/symbols/xjo.asx/ticks?appID=af5f4d73c1a54a33&adjustment=capital&fields=dc&range=20y"

# Headers for the request
headers = {
    "accept": "application/json",
    "referer": "https://www.marketindex.com.au/asx200",
    "sec-ch-ua": '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
}

try:
    # Send a GET request with headers
    response = requests.get(URL, headers=headers)

    # Check the response status
    if response.status_code == 200:
        # Parse the JSON response
        full_data = response.json()

        # Print the full data for debugging
        # print(json.dumps(full_data, indent=4))

        # Extract the "close" and "data" arrays if they exist
        filtered_data = {
            "date": full_data["ticks"]["date"],
            "data": full_data["ticks"]["close"],  # The closing price
        }

        # print the length of the "date" and "data" arrays
        # should be approx 20 yrs * 5 days * 52 weeks = 5200 days - minus some holidays....(5064)
        print(f"Length of 'date' array: {len(filtered_data['date'])}")
        print(f"Length of 'data' array: {len(filtered_data['data'])}")

        # compare the length of the "date" and "data" arrays
        if len(filtered_data["date"]) != len(filtered_data["data"]):
            print("Warning: The length of 'date' and 'data' arrays do not match.")

        # Save the filtered data to a file
        with open("filtered_data.json", "w", encoding="utf-8") as file:
            json.dump(filtered_data, file, indent=4)

        print("Filtered data saved to 'filtered_data.json'.")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
