import streamlit as st
import inspect
import textwrap


def show_code(demo):
    """Showing the code of the demo."""
    sourcelines, _ = inspect.getsourcelines(demo)
    st.code(textwrap.dedent("".join(sourcelines[1:])))


countryCODE = {
    "China": "CH",
    "United States": "US",
    "India": "IN",
    "Bosnia and Herzegovina": "BA",
    "Canada": "CA",
    "Bermuda": "BM",
    "Sudan": "SU",
    "Russia": "RS",
    "United Kingdom": "UK",
    "Hungary": "HU",
    "France": "FR",
    "Vietnam": "VM",
    "Madagascar": "MG",
    "Panama": "PM",
    "Mexico": "MX",
    "Jamaica": "JM",
    "South Africa": "ZA",
    "United Arab Emirates": "AE",
    "Australia": "AU",
    "South Korea": "KS",
    "Japan": "JA",
    "Central African Republic": "CT",
    "Nicaragua": "NU",
    "Denmark": "DA",
    "Germany": "GM",
    "Algeria": "AG",
    "Saudi Arabia": "SA",
    "Ethiopia": "ET",
    "Ukraine": "UP",
    "Ireland": "EI",
    "Israel": "IS",
    "Greece": "GR",
    "Italy": "IT",
    "West Bank": "WE",
    "Egypt": "EG",
    "Iran": "IR",
    "Lithuania": "LH",
    "Netherlands": "NL",
    "Poland": "PL",
    "Romania": "RO",
    "Democratic Republic of the Congo": "CD",
    "Afghanistan": "AF",
    "Grenada": "GJ",
    "Pakistan": "PK",
    "Portugal": "PO",
    "Czech Republic": "EZ",
    "Lebanon": "BL",
    "Turkey": "TU",
    "Qatar": "QA",
    "Bahamas": "BF",
    "Tunisia": "TS",
    "Barbados": "BB",
    "Sweden": "SW",
    "Colombia": "CO",
    "Belize": "BH",
    "Jersey": "JE",
    "Philippines": "RP",
    "Cambodia": "CB",
    "Angola": "AO",
    "Brazil": "BR",
    "Malaysia": "MY",
    "Botswana": "BC",
    "Ghana": "GH",
    "Spain": "SP",
    "Fiji": "FJ",
    "New Zealand": "NZ",
    "Samoa": "WS",
    "Luxembourg": "LU",
    "Somalia": "SO",
    "Sri Lanka": "CE",
    "Djibouti": "DJ",
    "Taiwan": "TW",
    "Kenya": "KE",
    "Ecuador": "EC",
    "Cuba": "CU",
    "Finland": "FI",
    "Liberia": "LI",
    "Syria": "SY",
    "Yemen": "YM",
    "Senegal": "SG",
    "Azerbaijan": "AJ",
    "Mozambique": "MZ",
    "Maldives": "MV",
    "Tuvalu": "TV",
    "Peru": "PE",
    "Armenia": "AM",
    "Indonesia": "ID",
    "Dominican Republic": "DR",
    "Burma (Myanmar)": "BU",
    "Zimbabwe": "ZI",
    "Bulgaria": "BG",
    "Libya": "LY",
    "Tanzania": "TZ",
    "Gabon": "GB",
    "Cyprus": "CY",
    "Iraq": "IZ",
    "Thailand": "TH",
    "French Guiana": "FG",
    "Nepal": "NP",
    "Mauritius": "MP",
    "Nigeria": "NG",
    "Ivory Coast": "IV",
    "Jordan": "JO",
    "Serbia": "RB",
    "Bolivia": "BO",
    "Belgium": "BE",
    "Saint Kitts and Nevis": "KN",
    "Uganda": "UG",
    "Croatia": "HR",
    "Kuwait": "KU",
    "Costa Rica": "CS",
    "Slovenia": "SI",
    "Venezuela": "VE",
    "Burkina Faso": "UV",
    "Hong Kong": "HK",
    "Antarctica": "AY",
    "Switzerland": "SZ",
    "Rwanda": "RW",
    "Gaza Strip": "GZ",
    "Lesotho": "LT",
    "Sierra Leone": "SL",
    "Trinidad and Tobago": "TD",
    "Bahrain": "BN",
    "Equatorial Guinea": "GQ",
    "Malta": "MT",
    "Norway": "NO",
    "Morocco": "MO",
    "Guyana": "GY",
    "Montenegro": "MJ",
    "Cape Verde": "CV",
    "Macedonia": "MK",
    "Turkmenistan": "TX",
    "Bangladesh": "BD",
    "Guinea-Bissau": "GG",
    "Guinea": "GV",
    "Kyrgyzstan": "KG",
    "Greenland": "GL",
    "Iceland": "IC",
    "Solomon Islands": "BP",
    "Uzbekistan": "UZ",
    "South Ossetia": "OS",
    "Oman": "MU",
    "Mali": "ML",
    "Congo (Brazzaville)": "CG",
    "New Caledonia": "NC",
    "Haiti": "HA",
    "Laos": "LA",
    "Uruguay": "UY",
    "Estonia": "EN",
    "Guatemala": "GT",
    "Malawi": "MI",
    "Argentina": "AR",
    "Tonga": "TO",
    "Latvia": "LG",
    "Moldova": "MD",
    "Kazakhstan": "KZ",
    "Bhutan": "BT",
    "Brunei": "BX",
    "Antigua and Barbuda": "AC",
    "El Salvador": "SV",
    "Albania": "AL",
    "Gibraltar": "GI",
    "Namibia": "WA",
    "Monaco": "MC",
    "Niger": "RN",
    "Swaziland": "WZ",
    "Papua New Guinea": "PP",
    "Puerto Rico": "RQ",
    "Honduras": "HO",
    "Curacao": "CW",
    "U.S. Virgin Islands": "VQ",
    "Cayman Islands": "CJ",
}


def Eurostats_preprocessing():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    #### Cleaning the monthly asylum data
    handle = pd.read_table("../data/asylum_monthly.tsv")
    asyl_monthly = handle.copy()
    # asyl_monthly
    deconstructed_first_column = [
        line.split(",") for line in asyl_monthly[asyl_monthly.columns[0]]
    ]
    # deconstructed_first_column
    colum1_list = asyl_monthly.columns[0].split(",")
    # colum1_list
    asyl_monthly = asyl_monthly.drop(asyl_monthly.columns[0], axis=1)
    # asyl_monthly
    first_column = pd.DataFrame(deconstructed_first_column, columns=colum1_list)
    # first_column
    first_column = first_column.drop(["unit"], axis=1)

    monthly_asylum = pd.concat([first_column, asyl_monthly], axis=1)
    monthly_asylum = monthly_asylum.rename(
        columns={"citizen": "from", monthly_asylum.columns[4]: "to"}
    )
    # Selecting all sexes
    monthly_asylum = monthly_asylum[monthly_asylum["sex"] == "T"]

    # Selecting all ages
    monthly_asylum = monthly_asylum[monthly_asylum["age"] == "TOTAL"]

    # Selecting only the first time applicants
    monthly_asylum = monthly_asylum[monthly_asylum["asyl_app"] == "NASY_APP"]

    # droping all these features as the only have one value now
    monthly_asylum = monthly_asylum.drop(["sex", "age", "asyl_app"], axis=1)

    # Reseting index now
    monthly_asylum = monthly_asylum.reset_index(drop=True)

    ### Here we define the best indexing format to tidly handle this data frame
    # We ['from','to'] the dyad values to the index
    monthly_asylum = monthly_asylum.set_index(["from", "to"])
    # We transpose it so the months columns are now the vertical index
    # and the dyads are accesible as horizontal indexes
    monthly_asylum = monthly_asylum.T
    monthly_asylum = monthly_asylum.rename_axis("date")
    # We convert the date index to datetime format
    monthly_asylum.index = monthly_asylum.index.str.strip()
    monthly_asylum.index = pd.to_datetime(
        monthly_asylum.index, format="%YM%m"
    ).strftime("%Y-%m")

    # Casting items to clean integers
    import re

    # apply strip() to all items in the data frame
    monthly_asylum = monthly_asylum.applymap(
        lambda x: x.strip() if isinstance(x, str) else x
    )

    monthly_asylum = monthly_asylum.replace(":", np.nan)
    monthly_asylum = monthly_asylum.replace("", np.nan)

    # replace all values like ': d' with no digits in it with np.nan
    monthly_asylum = monthly_asylum.applymap(
        lambda x: np.nan
        if isinstance(x, str) and not any(char.isdigit() for char in x)
        else x
    )
    # define a regular expression to remove non-numeric characters
    regex = re.compile(r"[^\d]")
    # apply the regular expression to the integer items
    monthly_asylum = monthly_asylum.applymap(
        lambda x: int(regex.sub("", x)) if isinstance(x, str) else x
    )

    # Reinvert the data frame to put it in the conventional chronological order.
    monthly_asylum = monthly_asylum[::-1]

    monthly_asylum.to_csv("../data/asylum_monthly_clean.csv")
    #### Extracting one dyad
    handle = pd.read_csv(
        "../data/asylum_monthly_clean.csv"
        # The following arguments are essential to correctly read the data frame
        ,
        index_col=0,
        header=[0, 1],
    )  # take the first to rows as headers/ horizontal multiindex

    asyl_apps = handle
    # This is how we would get a "Y" to be predicted for the VE-DE dyad, as it is a time series
    asyl_apps["VE"]["DE"]
    series = asyl_apps["VE"]["DE"].to_frame()
    series.name

    def get_dyad(asyl_df, origin, destination):
        """Funtion that returns a dataframe row for the given dyad
        pd.DataFrame    : df
        str             : origin
        str             : destination
        """
        dyad_series = asyl_df[origin][destination]
        # The actual reason for this function, to get a series where the name of the dyad can be storaged.
        dyad_series.name = f"{origin}->{destination}"
        return dyad_series

    get_dyad(asyl_apps, "VE", "DE")

    def show_asylum_apps(asyl_dyad) -> None:
        """Funtion that displays a seaborn graphic of the asylum applications from a given
        pd.Series : this way one passes the specific timeseries of the dyad
        """
        # Reformatting for seaborn
        dyad = asyl_dyad.to_frame()[::-1].reset_index()
        dyad.columns = ["date", "value"]
        # return dyad.info()
        # create line plot using seaborn
        sns.set_style("darkgrid")
        plt.figure(figsize=(25, 6))
        sns.lineplot(data=dyad, x="date", y="value")
        plt.xticks(rotation=60, ha="right")
        # set x-axis label and title
        plt.xlabel("Date")
        plt.ylabel("Asylum Applications")
        plt.title(f"Monthly asylum applications for {asyl_dyad.name}")
        plt.show()

    show_asylum_apps(get_dyad(asyl_apps, "VE", "DE"))
    show_asylum_apps(handle, "VE", "DE")
    ## Preprocessing the quarterly asylum recognitions
    handle = pd.read_table("../data/recognition_quarterly.tsv")

    asylum_recognition = handle
    deconstructed_first_column = [
        line.split(",") for line in asylum_recognition[asylum_recognition.columns[0]]
    ]
    colum1_list = asylum_recognition.columns[0].split(",")

    asylum_recognition = asylum_recognition.drop(asylum_recognition.columns[0], axis=1)
    first_column = pd.DataFrame(deconstructed_first_column, columns=colum1_list)
    first_column = first_column.drop(["unit"], axis=1)

    quarterly_recognition = pd.concat([first_column, asylum_recognition], axis=1)
    quarterly_recognition = quarterly_recognition.rename(
        columns={"citizen": "from", quarterly_recognition.columns[4]: "to"}
    )
    quarterly_recognition["decision"].value_counts()
    quarterly_recognition = quarterly_recognition[quarterly_recognition["sex"] == "T"]
    quarterly_recognition = quarterly_recognition[
        quarterly_recognition["age"] == "TOTAL"
    ]
    quarterly_recognition = quarterly_recognition[
        quarterly_recognition["decision"] != "GENCONV"
    ]
    quarterly_recognition = quarterly_recognition[
        quarterly_recognition["decision"] != "REJECTED"
    ]
    quarterly_recognition = quarterly_recognition[
        quarterly_recognition["decision"] != "SUB_PROT"
    ]
    quarterly_recognition = quarterly_recognition[
        quarterly_recognition["decision"] != "HUMSTAT"
    ]
    quarterly_recognition = quarterly_recognition.drop(["sex", "age"], axis=1)
    import re

    # apply strip() to all items in the data frame
    quarterly_recognition = quarterly_recognition.applymap(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    quarterly_recognition
    quarterly_recognition = quarterly_recognition.replace(":", np.nan)
    quarterly_recognition = quarterly_recognition.replace("", np.nan)
    quarterly_recognition
    # replace all values like ': d' with no digits in it with np.nan
    quarterly_recognition.iloc[:, 3:] = quarterly_recognition.iloc[:, 3:].applymap(
        lambda x: np.nan
        if isinstance(x, str) and not any(char.isdigit() for char in x)
        else x
    )
    # define a regular expression to remove non-numeric characters
    regex = re.compile(r"[^\d]")

    # apply the regular expression to the relevant items
    quarterly_recognition.iloc[:, 3:] = quarterly_recognition.iloc[:, 3:].applymap(
        lambda x: int(regex.sub("", x)) if isinstance(x, str) else x
    )
    quarterly_recognition
    quarterly_recognition.to_csv("../data/recognition_quarterly_clean.csv")
    ## Extract recognition rate table
    # quarterly recognition algorythm
    handle = pd.read_csv("../data/recognition_quarterly_clean.csv", index_col=0)
    qr_decisions = handle
    total_recognitions = qr_decisions[qr_decisions["decision"] == "TOTAL"].reset_index(
        drop=True
    )
    total_recognitions = total_recognitions.replace(0.0, 0.00000001)
    total_recognitions
    positive_recognitions = qr_decisions[
        qr_decisions["decision"] == "TOTAL_POS"
    ].reset_index(drop=True)
    positive_recognitions
    # preliminary template of the general recognition rate data frame
    recognition_rate = positive_recognitions.drop("decision", axis=1)
    recognition_rate.iloc[:, 2:] = round(
        (positive_recognitions.iloc[:, 3:] / total_recognitions.iloc[:, 3:]) * 100, 2
    )
    recognition_rate

    # We ['from','to'] the dyad values to the index
    recognition_rate = recognition_rate.set_index(["from", "to"])
    # We transpose it so the months columns are now the vertical index
    # and the dyads are accesible as horizontal indexes
    recognition_rate = recognition_rate.T
    recognition_rate = recognition_rate.rename_axis("date")
    # We convert the date index to datetime format
    recognition_rate.index = recognition_rate.index.str.strip()
    recognition_rate.index = pd.PeriodIndex(
        recognition_rate.index, freq="Q"
    ).to_timestamp()
    recognition_rate = recognition_rate[::-1]
    recognition_rate
    recognition_rate.to_csv("../data/recognition_rates.csv")
    ### Nexts steps: clock align the recognition rates to at least monthly data.
    rr = pd.read_csv(
        "../data/recognition_rates.csv", index_col=0, header=[0, 1], parse_dates=True
    )
    rr
    start_date = "2008-01-01"
    end_date = "2023-01-01"  # pd.to_datetime('today')  # Get the current month and year
    monthly_index = pd.date_range(
        start=start_date, end=end_date, freq=pd.offsets.MonthBegin()
    )
    rr = rr.reindex(monthly_index).interpolate(method="linear")
    rr.to_csv("../data/recognition_rates_interpolated.csv")
    dyad = get_dyad(recognition_rate, "VE", "DE")
    dyad

    pd.melt(dyad.iloc[:, 2:]).set_index("variable").index

    def dyad_recognition_timeseries(recognition_rates, origin, destination):
        dyad = get_dyad(recognition_rates, origin, destination)
        dyad_recognitionTS = pd.melt(
            dyad.iloc[:, 2:], var_name="quarter", value_name="recognition_rate"
        ).set_index("quarter")
        dyad_recognitionTS.index = dyad_recognitionTS.index.str.strip()
        dyad_recognitionTS.index = pd.PeriodIndex(
            dyad_recognitionTS.index, freq="Q"
        ).to_timestamp()
        return dyad_recognitionTS

    dyad_recognition_timeseries(recognition_rate, "VE", "DE")
    ## Atomatize lineplotting for different dyads
    handle = pd.read_csv("../data/recognition_rates.csv", index_col=0)
    recognition_rates = handle
    recognition_rates
    VE_DE_recognitions = dyad_recognition_timeseries(recognition_rates, "VE", "DE")
    VE_DE_recognitions

    def plot_recognition_rates(recognition_rates, origin, destination):
        dyad_TS = dyad_recognition_timeseries(recognition_rates, origin, destination)
        sns.set_style("darkgrid")
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=dyad_TS, x=dyad_TS.index, y="recognition_rate")

        # set x-axis label and title
        plt.xlabel("Date")
        plt.ylabel("recognition rate")
        plt.title(
            f"Quarterly recognition rates from {origin} asylum applications in {destination}"
        )

        plt.show()

    plot_recognition_rates(recognition_rates, "VE", "DE")
    plot_recognition_rates(recognition_rates, "SY", "DE")
    plot_recognition_rates(recognition_rates, "UA", "EU27_2020")


def GDELT_preprocessing():
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
                csv_filename = zip_ref.namelist()[
                    0
                ]  # Assuming only one file in the zip
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
