import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_csv_files(user_names):
    # Create an empty DataFrame to store the combined results
    combined_df = pd.DataFrame()

    # Concatenate multiple CSV files into a single DataFrame
    for user_name in user_names:
        csv_file_path = f"./data_collected/{user_name}/feedback/results.csv"
        if os.path.isfile(csv_file_path):
            df = pd.read_csv(csv_file_path)
            combined_df = pd.concat([combined_df, df])

    # Calculate average scores for each combination of users, alpha values, and methods
    average_scores_df = (
        combined_df.groupby(["Method", "Alpha"])["Score"].mean().reset_index()
    )

    return average_scores_df


def create_bar_chart(average_scores_df, output_folder):
    # Create a single bar chart with all methods and alpha values overlaying each other
    methods = average_scores_df["Method"].unique()
    alpha_values = average_scores_df["Alpha"].unique()

    fig, ax = plt.subplots(figsize=(10, 5))

    width = 0.2
    x_positions = np.arange(len(alpha_values))

    for i, method in enumerate(methods):
        subset_df = average_scores_df[average_scores_df["Method"] == method]

        # Align scores with all alpha values using the align function
        aligned_scores = subset_df.set_index("Alpha")["Score"].align(
            pd.Series(0, index=alpha_values)
        )[0].values

        ax.bar(
            x_positions + i * width,
            aligned_scores,
            width=width,
            align="center",
            label=f"Method '{method}'",
        )

    ax.set_xticks(x_positions + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(alpha_values)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Score")
    ax.set_title("Average Scores for Different Methods and Alpha Values")
    ax.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "score_bar_chart_feedback.png"))
    plt.close()


if __name__ == "__main__":
    user_names = [
        "Abdelrahman",
        "Abdelrahman_2",
        "Mahmoud",
    ]  # Replace with the list of user names
    output_folder = (
        "./data_collected/average_scores"  # Replace with the desired output folder path
    )

    # Process CSV files and calculate average scores
    average_scores_df = process_csv_files(user_names)

    # Save the average scores to a new CSV file
    average_scores_df.to_csv(
        os.path.join(output_folder, "average_scores_feedback.csv"), index=False
    )

    # Create a single bar chart with all methods and alpha values overlaying each other
    create_bar_chart(average_scores_df, output_folder)

