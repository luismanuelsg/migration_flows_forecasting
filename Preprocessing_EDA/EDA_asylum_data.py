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
monthly_asylum.index = pd.to_datetime(monthly_asylum.index, format="%YM%m").strftime(
    "%Y-%m"
)

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
quarterly_recognition = quarterly_recognition[quarterly_recognition["age"] == "TOTAL"]
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
recognition_rate.index = pd.PeriodIndex(recognition_rate.index, freq="Q").to_timestamp()
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
