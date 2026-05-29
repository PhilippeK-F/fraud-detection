import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def extract():
    os.environ["philippekirstetter"] = os.getenv("philippekirstetter", "")
    os.environ["KGAT_c7a87ae70434bc51ffbee7e44b5d47ad"]      = os.getenv("KGAT_c7a87ae70434bc51ffbee7e44b5d47ad", "")

    import kaggle

    print("[extract] Telechargement dataset fraude bancaire...")
    try:
        kaggle.api.dataset_download_files(
            "mlg-ulb/creditcardfraud",
            path="data/raw",
            unzip=True
        )
        print("[extract] OK -> data/raw/creditcard.csv")
    except Exception as e:
        print("[extract] Erreur Kaggle : " + str(e))
        return None

    df = pd.read_csv("data/raw/creditcard.csv")
    print("[extract] " + str(len(df)) + " transactions chargees")
    print("[extract] Colonnes : " + str(list(df.columns)))
    print("[extract] Fraudes : " + str(df["Class"].sum()) + " (" + str(round(df["Class"].mean()*100, 3)) + "%)")
    return df

if __name__ == "__main__":
    extract()