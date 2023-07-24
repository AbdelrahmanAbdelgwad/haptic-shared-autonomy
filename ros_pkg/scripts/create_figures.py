import pandas as pd
import matplotlib.pyplot as plt

user_name = "Abdelrahman"

# Read the CSV file into a DataFrame
results_df = pd.read_csv(f"./data_collected/{user_name}/feedback/results.csv")

# Create bar charts for each method and alpha value
methods = results_df["Method"].unique()
alpha_values = results_df["Alpha"].unique()

for method in methods:
    for alpha in alpha_values:
        # Filter the DataFrame for the specific method and alpha value
        subset_df = results_df[
            (results_df["Method"] == method) & (results_df["Alpha"] == alpha)
        ]

        # Create the bar chart
        plt.figure(figsize=(10, 5))
        plt.bar(subset_df["Alpha"], subset_df["Real Score"], color="blue", width=0.4)
        plt.xlabel("Alpha")
        plt.ylabel("Real Score")
        plt.title(f"Real Scores for Method '{method}' with Alpha '{alpha}'")
        plt.savefig(
            f"./data_collected/{user_name}/feedback/{method}/{method}_alpha_{alpha}_real_score_bar_chart.png"
        )
        plt.close()
