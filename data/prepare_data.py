import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# ------------------------------
# 1. Kaggle-Daten laden
# ------------------------------
def fetch_accidents():
    """
    Lädt das US-Accidents-Dataset von Kaggle über kagglehub (Version >=0.4.0)
    """
    df = kagglehub.dataset_load(
        adapter=KaggleDatasetAdapter.PANDAS,
        handle="sobhanmoosavi/us-accidents",
        path="US_Accidents_March23.csv",
        pandas_kwargs={
            "usecols": [
                "ID",
                "Severity",
                "Start_Time",
                "State",
                "City",
                "Weather_Condition",
                "Start_Lat",
                "Start_Lng",
                "Temperature(F)",
                "Visibility(mi)",
                "Crossing",
                "Junction",
                "Traffic_Signal"
            ]
        }
    )
    return df



# ------------------------------
# 2. Daten aufbereiten
# ------------------------------
def preprocess_data(df, sample_frac=None):
    """
    Bereinigt das Dataset:
    - Start_Time korrekt parsen
    - State-Abkürzungen zu vollen Namen mappen
    - Optional: Sampling für kleinere Dateien
    """
    # Start_Time robust parsen
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce", utc=True)

    # State-Abkürzungen zu Namen
    us_state_to_abbrev = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
        "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
        "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
        "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
        "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
        "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
        "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
        "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
        "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
        "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
        "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia"
    }
    df["State"] = df["State"].map(us_state_to_abbrev).fillna(df["State"])

    # Optional: Sampling
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=42)

    return df

# ------------------------------
# 3. Daten speichern
# ------------------------------
def save_data(df, filename="data/us_accidents_prepared.csv"):
    """
    Speichert das vorbereitete Dataset als CSV.
    """
    df.to_csv(filename, index=False)
    print(f"✅ Daten gespeichert: {filename} | {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

# ------------------------------
# 4. Alles zusammen ausführen
# ------------------------------
if __name__ == "__main__":
    print("🚀 Lade Rohdaten von Kaggle...")
    df_raw = fetch_accidents()

    print("⚙️  Daten aufbereiten...")
    # Für schnelle Tests kannst du sample_frac=0.05 nehmen
    df_clean = preprocess_data(df_raw, sample_frac=0.05)

    print("💾 Speichere CSV...")
    save_data(df_clean)
