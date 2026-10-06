import pandas as pd

# TODO: add make folder part 

fs_df = pd.read_csv("data/oe/raw/Fitness_Scores_Imputed.tsv", sep="\t", header=0, index_col="UID")

print(fs_df)

fs_df = fs_df.filter(like="_Avg_log2_Fitness_Score")
fs_df.columns = fs_df.columns.str.replace("_Avg_log2_Fitness_Score", "")
print(fs_df.shape)
print(fs_df)  # expect 15 columns
fs_df.to_csv("data/oe/prepped/fitness_scores.csv", sep="\t", header=True, index=True)

# Long format: one row per gene-strain pair and drop the NaNs
long_fs_df = fs_df.reset_index().melt(id_vars="UID", var_name="strain", value_name="fitness_score").dropna() 
# TODO: add a file to show what genes per strain were dropped for being NaN
print(long_fs_df)
long_fs_df.to_csv("data/oe/prepped/long_fitness_scores.csv", sep="\t", header=True, index=False)



