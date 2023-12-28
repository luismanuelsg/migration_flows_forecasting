import zipfile
import os
import urllib.request
from datetime import datetime, timedelta
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def eventdf_FE(eventdf, CAMEOtags):
    """Filter the giant day GDELT event data to its relevant columns"""
    df = eventdf[
        [
            "SQLDATE",
            "IsRootEvent",
            "EventCode"
            # ,'QuadClass'           Excluded as we build the event indices proposed by the paper
            # ,'GoldsteinScale'
            ,
            "ActionGeo_CountryCode"
            # ,'SOURCEURL'           As we are going to agregate all these for a month, it is not relevant anymore to track it back
        ]
    ].copy()
    # Extract only root events as done in the paper
    df = df[df["IsRootEvent"] == 1]
    df = df.drop("IsRootEvent", axis=1)
    # Extract only the relevant events
    df = df[df["EventCode"].isin(CAMEOtags)]
    # drop NaNs, as they are mostly in the CountryCode
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df.rename(
        columns={
            "SQLDATE": "date",
            "EventCode": "code",
            "ActionGeo_CountryCode": "country",
        }
    )
    return df


def get_category_weights(eventdf, CAMEO_weights):
    """Get the aggregated weights for each of the 5 event categories for each country
    as described in the paper, for a day"""
    df = eventdf_FE(eventdf, CAMEO_weights.index.tolist())

    # Aggregate them by country and event-type-code
    # and get the quantity of the single event-types for in each country
    df = df.groupby(["country", "code"]).size()
    df = pd.DataFrame(df).reset_index().rename(columns={0: "count"})

    # Now we are associating each event-type to their corresponding
    # weights and labels describe in the CAMEO_weights reference table
    df["category"] = pd.NA
    for i, row in df.loc[:, ["code"]].iterrows():
        weight = CAMEO_weights.loc[row["code"]]["weight"]
        count = df["count"].iloc[i]
        df["count"].iloc[i] = count * weight / 3
        df["category"].iloc[i] = CAMEO_weights.loc[row["code"]]["label"]

    # Done this, we don't need the code column anymore
    # so we drop it, an agreggate by country and category
    df = df.drop("code", axis=1).groupby(["country", "category"]).sum()
    return df


def extract_csv_from_zip(url, target_folder):
    try:
        # Download the zip file
        zip_file_path = os.path.join(target_folder, "data.zip")
        urllib.request.urlretrieve(url, zip_file_path)

        # Extract the CSV file from the zip
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            csv_filename = zip_ref.namelist()[0]  # Assuming only one file in the zip
            csv_file_path = os.path.join(target_folder, csv_filename)
            zip_ref.extract(csv_filename, target_folder)

        # Delete the zip file
        os.remove(zip_file_path)

        return csv_file_path

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None


def process_gdelt_data(
    start_date, end_date, target_folder, CAMEO_weights, header, verbose=False
):
    """This function:
    1.Scrapes the daily events-zip files from the GDELT web
    2.Extracts the csvs from the zips
    3.Filters, aggregates and preprocess the data in the csv's
    4.Leaves only the already preprocessed csv's in the hard disk
      ready to be aggregated per month so they can be added in the model
    """
    delta = timedelta(days=1)  # Increment of one day
    missing_dates = []

    while end_date >= start_date:
        sqldate = end_date.strftime("%Y%m%d")
        url = f"http://data.gdeltproject.org/events/{sqldate}.export.CSV.zip"
        csv_file_path = extract_csv_from_zip(url, target_folder)

        # Checking if its a missing date
        if csv_file_path == None:
            print(f"URL for SQLDATE {sqldate} does not exist. Skipping.")
            missing_dates.append(sqldate)
            end_date -= delta
            continue

        elif verbose:
            print(f"File for SQLDATE {sqldate} extracted successfully.")

        # Read CSV file into DataFrame
        df = pd.read_table(csv_file_path, names=header, dtype={"EventCode": str})

        # Process the DataFrame as needed
        df = get_category_weights(df, CAMEO_weights)

        # Save DataFrame as CSV
        output_csv_path = os.path.join(target_folder, f"{sqldate}.csv")
        df.to_csv(output_csv_path)

        # Delete the intermediate CSV file
        os.remove(csv_file_path)

        if verbose:
            print(f"Processed data saved as {output_csv_path}")

        end_date -= delta

    return missing_dates


if __name__ == "__main__":
    header = "GLOBALEVENTID   SQLDATE MonthYear   Year    FractionDate	Actor1Code	Actor1Name	Actor1CountryCode	Actor1KnownGroupCode	Actor1EthnicCode	Actor1Religion1Code	Actor1Religion2Code	Actor1Type1Code	Actor1Type2Code	Actor1Type3Code	Actor2Code	Actor2Name	Actor2CountryCode	Actor2KnownGroupCode	Actor2EthnicCode	Actor2Religion1Code	Actor2Religion2Code	Actor2Type1Code	Actor2Type2Code	Actor2Type3Code	IsRootEvent	EventCode	EventBaseCode	EventRootCode	QuadClass	GoldsteinScale	NumMentions	NumSources	NumArticles	AvgTone	Actor1Geo_Type	Actor1Geo_FullName	Actor1Geo_CountryCode	Actor1Geo_ADM1Code	Actor1Geo_Lat	Actor1Geo_Long	Actor1Geo_FeatureID	Actor2Geo_Type	Actor2Geo_FullName	Actor2Geo_CountryCode	Actor2Geo_ADM1Code	Actor2Geo_Lat	Actor2Geo_Long	Actor2Geo_FeatureID	ActionGeo_Type	ActionGeo_FullName	ActionGeo_CountryCode	ActionGeo_ADM1Code	ActionGeo_Lat	ActionGeo_Long	ActionGeo_FeatureID	DATEADDED	SOURCEURL"
    header = header.split()
    CAMEO_weights = pd.read_csv(
        "../data/CAMEO_weights.csv", index_col=0, dtype={"code": str}
    )
    CAMEO_weights = CAMEO_weights.set_index("code").drop("description", axis=1)

    start_date = datetime.strptime("20150101", "%Y%m%d")
    # end_date = datetime.now() - timedelta(days=1)   # Uses the date of yesterday as the date
    end_date = datetime.strptime("20230323", "%Y%m%d")
    target_folder = "../data/gdelt_optimized/"

    process_gdelt_data(
        start_date, end_date, target_folder, CAMEO_weights, header, verbose=False
    )
