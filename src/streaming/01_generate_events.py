import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "NoShowAppointments.csv"
STREAM_INPUT_DIR = BASE_DIR / "data" / "stream_input"

STREAM_INPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_event(record):
    scheduled_dt = datetime.now()
    appointment_dt = scheduled_dt + timedelta(days=random.randint(0, 30))

    return {
        "PatientId": float(record["PatientId"]),
        "AppointmentID": int(record["AppointmentID"]),
        "Gender": record["Gender"],
        "ScheduledDay": scheduled_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "AppointmentDay": appointment_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Age": int(record["Age"]),
        "Neighbourhood": record["Neighbourhood"],
        "Scholarship": int(record["Scholarship"]),
        "Hipertension": int(record["Hipertension"]),
        "Diabetes": int(record["Diabetes"]),
        "Alcoholism": int(record["Alcoholism"]),
        "Handcap": int(record["Handcap"]),
        "SMS_received": int(record["SMS_received"]),
        "No-show": record["No-show"],
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    print("Generating appointment events...")
    print(f"Writing to: {STREAM_INPUT_DIR}")

    for i in range(20):
        record = df.sample(1).iloc[0].to_dict()
        event = make_event(record)

        output_file = STREAM_INPUT_DIR / f"appointment_event_{int(time.time())}_{i}.json"

        with open(output_file, "w") as f:
            json.dump(event, f)

        print(f"Wrote: {output_file.name}")
        time.sleep(2)

    print("Done generating events.")