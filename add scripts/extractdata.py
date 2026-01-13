import pandas as pd

# =========================
# 1. FILE PATHS
# =========================
file_repisgt = r"C:/Users/kevin/OneDrive/Desktop/projects-kevin/Crsipr experimenting folder/eg_reg_on_target.repisgt"
file_rsgt = r"C:/Users/kevin/OneDrive/Desktop/projects-kevin/Crsipr experimenting folder/eg_reg_on_target_seq.rsgt"

# =========================
# 2. READ FILES (TAB-SEPARATED)
# =========================
df1 = pd.read_csv(file_repisgt, sep="\t", header=None)
df2 = pd.read_csv(file_rsgt, sep="\t", header=None)

print(f"Rows in .repigst: {len(df1)}")
print(f"Rows in .rsgt: {len(df2)}")

# =========================
# 3. COMBINE DATAFRAMES
# =========================
combined_df = pd.concat([df1, df2], ignore_index=True)
print("Total rows after combining:", len(combined_df))

# =========================
# 4. SAVE COMBINED CSV
# =========================
combined_csv_path = r"C:/Users/kevin/OneDrive/Desktop/projects-kevin/Crsipr experimenting folder/combined_dataset.csv"
combined_df.to_csv(combined_csv_path, index=False)
print(f"Combined dataset saved at: {combined_csv_path}")
