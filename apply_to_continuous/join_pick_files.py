#%%
import pandas as pd
df1 = pd.read_csv("../../newer/applied_results/picks_0325_0329_CNNproc.csv")

df2 = pd.read_csv("../../newer/applied_results/picks_0330_0403YGC.csv")

df3 = pd.read_csv("../../newer/applied_results/YGC.0403.csv")

#%%

df_post30 = pd.concat([df2, df3])
df_post30 = df_post30.sort_values("arrival_time")
# %%

df_all = pd.concat([df1, df_post30])
df_all = df_all.sort_values("arrival_time")
# %%
df_all.to_csv("../../newer/applied_results/all_picks_0325_0403_CNNproc.csv", index=False)
# %%
