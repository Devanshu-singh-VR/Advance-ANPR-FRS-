import requests
import csv
from datetime import datetime
import pytz

POWER_AUTOMATE_URL_ENTRY = "https://prod-03.centralindia.logic.azure.com:443/workflows/6142c53062964c50831cb589fb08dc66/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=KAfWkwHf5S2xifDKpojIdRdvbM2N6Y8reMp0wFyayrk"

def send_trigger_request(number_plate, email_id, timestamp):
    data = {
        "number_plate": number_plate,
        "email_id": email_id,
        "timestamp": timestamp
    }
    try:
        response = requests.post(POWER_AUTOMATE_URL_ENTRY, json=data)
        response.raise_for_status()
        return response.status_code, response.content
    except requests.exceptions.RequestException as e:
        return None, str(e)

def convert_to_ist(utc_datetime):
    utc_time = pytz.utc.localize(utc_datetime)
    ist_time = utc_time.astimezone(pytz.timezone('Asia/Kolkata'))
    return ist_time.strftime("%Y-%m-%d %H:%M:%S")

def main():
    csv_file = "entry_exit_alert.csv"  # Replace with the actual file path

    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            number_plate = row["number_plate"]
            email_id = row["email_id"]
            current_utc_time = datetime.utcnow()
            timestamp_ist = convert_to_ist(current_utc_time)

            status_code, response_content = send_trigger_request(number_plate, email_id, timestamp_ist)
            if status_code == 200:
                print(f"Email triggered for Car No. {number_plate} at {timestamp_ist} IST")
            else:
                print(f" triggering email for Car No. {number_plate}")
                if response_content:
                    print(f"Response Content: {response_content}")

if __name__ == "__main__":
    main()
