#%%
import pandas as pd

pick_file = "../../newer/applied_results/picks_0325_0330B945_CNNproc.csv"
pick_df = pd.read_csv(pick_file)

#%%
epoch_time = 1396137600

last_day_df = pick_df[pick_df.arrival_time >= epoch_time]
# %%
from obspy.core.utcdatetime import UTCDateTime

epoch_time_obs = UTCDateTime("20140330") - UTCDateTime("19700101")
# %%
earlier_df =  pick_df[pick_df.arrival_time < epoch_time]
# %%

earlier_df.to_csv("../../newer/applied_results/picks_0325_0329_CNNproc.csv", index=False)
last_day_df.to_csv("../../newer/applied_results/picks_0330partial_B945_CNNproc.csv", index=False)
# %%
