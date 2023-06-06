
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# print(pd.__version__)


# df["AlphaNumeric"] = df["alpha"].apply(lambda x: int(x[2:-1]))
# print table
# print(df.info())
# print(df[["filename", "AccuracyPersonForcedZoom"]])

# plot

# df_data = df.loc[(df["Attaques"] == "shapeshifter")
#                  | (df["Attaques"] == "témoin")]
# print(df_data[["Attaques2", "FocaleNumeric", "AccuracyPerson", "AccuracyStopSign"]])

#  seaborn
# sns.jointplot(data=df_data, x="FocaleNumeric",
#               y="AccuracyStopSign", hue="Attaques2")
# plt.title("Sttop sign accuracy /  focale")
# plt.show()


# plt old school
# fig, ax = plt.subplots()
# plt.scatter(df_data["LongitudeNumeric"],
#             df_data["AccuracyCar"])
# for i, txt in enumerate(df_data["Attaques2"]):
#     ax.annotate(
#         txt, (df_data["LongitudeNumeric"].iloc[i], df_data["AccuracyPerson"].iloc[i]-0.1*np.random.random()))
# plt.title("shapeshifter and témoin")

# plotly
# fig = px.scatter(df_data, x="LongitudeNumeric", y="FocaleNumeric",
#                  text="Attaques2", log_x=True, size_max=100, size=list(df_data["AccuracyStopSign"]), color="alpha")
# fig.update_traces(textposition='top center')
# fig.update_layout(title_text='ShapeShifter', title_x=0.5)
# fig.show()


# all zooms 
# df_data = df.loc[(df["Attaques"] == "shapeshifter")
#                  | (df["Attaques"] == "témoin")]

def add_all_zooms(column_name, df):
        '''
        This unfolds the chose zoom accuracies into several columns in the dataframe. 
        '''

        df[[f"{column_name}1",
                f"{column_name}2",
                f"{column_name}3",
                f"{column_name}4",
                f"{column_name}5",
                f"{column_name}6"]] = df.apply(lambda x:pd.Series(x[f"{column_name}"][1:-1].split(",")), axis=1)
        df[[f"{column_name}1",
                f"{column_name}2",
                f"{column_name}3",
                f"{column_name}4",
                f"{column_name}5",
                f"{column_name}6"]] = df[[f"{column_name}1",
                                                f"{column_name}2",
                                                f"{column_name}3",
                                                f"{column_name}4",
                                                f"{column_name}5",
                                                f"{column_name}6"]].apply(lambda x:pd.Series([float(x[i])  if ("No detection" not in x[i]) else None for i in range(len(x))]), axis=1)
        return df



# read from memory
df = pd.read_excel("./results.xlsx", engine='openpyxl')

# data cleaning
# drop the string attributes when no detection
df = df.replace(to_replace="No detection", value=None)
# Focal string to int
df["FocaleNumeric"] = df["Focale"].apply(lambda x: int(x[1:-2]))
# Longitude string to float
df["LongitudeNumeric"] = df["Longitude"].apply(lambda x: float(x[:-1]))

# unfolding zoom accuracies
df = add_all_zooms("AccuracyCarForcedZoom", df)
df = add_all_zooms("AccuracyPersonForcedZoom", df)
df = add_all_zooms("AccuracyStopSignForcedZoom", df)

print(df.info())
# saving to excel 
df.to_excel("final_results.xlsx")