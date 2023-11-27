from matplotlib.font_manager import FontProperties
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import chi2_contingency
from collections import Counter
from sklearn.model_selection import train_test_split
from libs.data_preprocessing.raw_data_preprocessor import (
    RawDataPreprocessor,
    PostRawDataPreprocessor,
)
from libs.data_preprocessing.dataset_preprocessor import (
    LevelDatasetPreprocessor,
    ReasonDetailDatasetPreprocessor,
    SystemDatasetPreprocessor,
    ReasonDatasetPreprocessor,
    PersonDatasetPreprocessor,
)

sns.set_palette("deep")

matplotlib.rcParams["font.family"] = ["Heiti TC"]
font_prop = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")

GROUND_TRUTH_RAW_DATA = "raw_data/ground_truth.csv"
POST_GROUND_TRUTH_RAW_DATA = "raw_data/post_ground_truth.csv"
TEST_DATA = "elk_test.csv"


def main():
    raw_data_preprocessor: RawDataPreprocessor = RawDataPreprocessor(
        GROUND_TRUTH_RAW_DATA
    )

    processed_df: pd.DataFrame = raw_data_preprocessor.get_processed_dataframe()

    # level data
    [level_features_df, level_label_df] = LevelDatasetPreprocessor.process(processed_df)
    [
        _,
        reason_detail_label_df,
    ] = ReasonDetailDatasetPreprocessor.process(processed_df)
    [_, system_label_df] = SystemDatasetPreprocessor.process(processed_df)
    [_, reason_label_df] = ReasonDatasetPreprocessor.process(processed_df)
    [_, person_label_df] = PersonDatasetPreprocessor.process(processed_df)

    # Split into train and test
    level_features_df.to_csv("clean_data/level_features.csv")
    level_label_df.to_csv("clean_data/level_label.csv")
    reason_detail_label_df.to_csv("clean_data/reason_detail_label.csv")
    system_label_df.to_csv("clean_data/system_label.csv")
    reason_label_df.to_csv("clean_data/reason_label.csv")
    person_label_df.to_csv("clean_data/person_label.csv")


def test():
    raw_data_preprocessor: RawDataPreprocessor = RawDataPreprocessor(
        GROUND_TRUTH_RAW_DATA
    )
    processed_df: pd.DataFrame = raw_data_preprocessor.get_processed_dataframe()
    test_df: pd.DataFrame = pd.read_csv(TEST_DATA)
    level_dataset_preprocessor = LevelDatasetPreprocessor()
    system_dataset_preprocessor = SystemDatasetPreprocessor()
    reason_detail_dataset_preprocessor = ReasonDetailDatasetPreprocessor()
    reason_dataset_preprocessor = ReasonDatasetPreprocessor()

    _, _ = level_dataset_preprocessor.process(processed_df)
    _, _ = reason_detail_dataset_preprocessor.process(processed_df)
    _, _ = system_dataset_preprocessor.process(processed_df)
    _, _ = reason_dataset_preprocessor.process(processed_df)

    level_test_label_df: pd.DataFrame = pd.read_csv("test/level_test_label.csv")
    system_test_label_df: pd.DataFrame = pd.read_csv("test/system_test_label.csv")
    reason_test_label_df: pd.DataFrame = pd.read_csv("test/reason_test_label.csv")
    reason_detail_test_label_df: pd.DataFrame = pd.read_csv(
        "test/reason_detail_test_label.csv"
    )

    level_test_label_df = level_dataset_preprocessor.inverse_process(
        level_test_label_df
    )
    system_test_label_df = system_dataset_preprocessor.inverse_process(
        system_test_label_df
    )
    reason_test_label_df = reason_dataset_preprocessor.inverse_process(
        reason_test_label_df
    )
    reason_detail_test_label_df = reason_detail_dataset_preprocessor.inverse_process(
        reason_detail_test_label_df
    )

    test_df["level_ai"] = level_test_label_df
    test_df["system_ai"] = system_test_label_df
    test_df["reason_ai"] = reason_test_label_df
    test_df["reason_detail_ai"] = reason_detail_test_label_df

    test_df = test_df[~test_df["message_detail"].str.contains("查無")]

    test_df.to_csv("elk_test.csv")

    print()


def split_dataframe():
    raw_data_preprocessor: RawDataPreprocessor = RawDataPreprocessor(
        GROUND_TRUTH_RAW_DATA
    )
    processed_df: pd.DataFrame = raw_data_preprocessor.get_processed_dataframe()

    first_half_processed_df, second_half_processed_df = train_test_split(
        processed_df, test_size=0.5
    )

    first_half_processed_df.to_csv("clean_data/halt.csv", index=False)


class ExploratoryDataAnalyzer:
    def __init__(self):
        super().__init__()

    def analyze_text_length(self, data: pd.DataFrame):
        # Checking for missing or non-string values in 'message' and 'message_detail'
        missing_values = data[["message", "message_detail"]].isnull().sum()

        # Replace NaNs with empty strings and recompute the lengths
        data["message"] = data["message"].fillna("")
        data["message_detail"] = data["message_detail"].fillna("")
        data["message_length"] = data["message"].apply(len)
        data["message_detail_length"] = data["message_detail"].apply(len)

        # Redo the plotting of the distribution of text lengths
        plt.figure(figsize=(15, 6))

        # Plot for 'message'
        plt.subplot(1, 2, 1)
        sns.histplot(data["message_length"], bins=50, kde=True)
        plt.title("Distribution of Message Length")
        plt.xlabel("Length of Message")
        plt.ylabel("Frequency")

        # Plot for 'message_detail'
        plt.subplot(1, 2, 2)
        sns.histplot(data["message_detail_length"], bins=50, kde=True)
        plt.title("Distribution of Message Detail Length")
        plt.xlabel("Length of Message Detail")
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig("analysis/text_length.jpg", dpi=600)

        # Descriptive statistics for text length
        text_length_stats = data[["message_length", "message_detail_length"]].describe()
        missing_values, text_length_stats

    def analyze_label_distribution(self, data: pd.DataFrame):
        def plot_label_distribution_bar_charts(
            columns, column_labels, data, chart_size=(10, 6)
        ):
            for column, column_label in zip(columns, column_labels):
                plt.figure(figsize=chart_size)
                label_counts = data[column].value_counts()
                label_counts = label_counts.reindex(column_label, fill_value=0)
                sns.barplot(
                    x=label_counts.values, y=label_counts.index, hue=label_counts.index
                )
                plt.title(
                    f"Distribution of Labels in {column}", fontproperties=font_prop
                )
                plt.xlabel("Count", fontproperties=font_prop)
                plt.ylabel("Labels", fontproperties=font_prop)

                plt.savefig(f"analysis/label_distribution{column}.jpg", dpi=600)

        columns_to_visualize = ["level", "system", "person", "reason", "reason_detail"]
        column_labels = [
            LevelDatasetPreprocessor.label_names,
            SystemDatasetPreprocessor.label_names,
            PersonDatasetPreprocessor.label_names,
            ReasonDatasetPreprocessor.label_names,
            ReasonDetailDatasetPreprocessor.label_names,
        ]

        plot_label_distribution_bar_charts(columns_to_visualize, column_labels, data)

    def analyze_host_name(self, data: pd.DataFrame):
        # Visualizing relationships between 'hostname' and other categorical variables ('system', 'person', 'reason')
        # Function to create a heatmap for categorical variables
        def plot_categorical_heatmap(df, x_var, y_var, title):
            contingency_table = pd.crosstab(df[x_var], df[y_var])
            plt.figure(figsize=(15, 10))
            sns.heatmap(contingency_table, annot=False, cmap="viridis", fmt="g")
            plt.title(title)
            plt.xlabel(y_var)
            plt.ylabel(x_var)

        # Heatmap for 'hostname' vs 'system'
        plot_categorical_heatmap(
            data, "hostname", "system", "Heatmap of Hostname vs System"
        )

        # Heatmap for 'hostname' vs 'person'
        plot_categorical_heatmap(
            data, "hostname", "person", "Heatmap of Hostname vs Person"
        )

        # Heatmap for 'hostname' vs 'reason'
        plot_categorical_heatmap(
            data, "hostname", "reason", "Heatmap of Hostname vs Reason"
        )

    def analyze_token_occurance(self, data: pd.DataFrame):
        def tokenize(text):
            # Splitting by non-alphabetic characters for simplicity
            return re.findall(r"\b\w+\b", text.lower())

        # Tokenizing 'message' and 'message_detail'
        data["message_tokens"] = data["message"].apply(tokenize)
        data["message_detail_tokens"] = data["message_detail"].apply(tokenize)

        # Counting the frequency of tokens
        message_token_freq = Counter(
            [token for sublist in data["message_tokens"] for token in sublist]
        )
        message_detail_token_freq = Counter(
            [token for sublist in data["message_detail_tokens"] for token in sublist]
        )

        # Most common tokens in 'message' and 'message_detail'
        top_10_message_tokens = message_token_freq.most_common(10)
        top_10_message_detail_tokens = message_detail_token_freq.most_common(10)

        top_10_message_tokens, top_10_message_detail_tokens

        # Plotting the top 10 tokens in 'message' and 'message_detail'

        plt.figure(figsize=(15, 10))

        # Top 10 tokens in 'message'
        plt.subplot(2, 1, 1)
        sns.barplot(
            x=[token for token, count in top_10_message_tokens],
            y=[count for token, count in top_10_message_tokens],
        )
        plt.title("Top 10 Tokens in Message")
        plt.xlabel("Tokens")
        plt.ylabel("Frequency")

        # Top 10 tokens in 'message_detail'
        plt.subplot(2, 1, 2)
        sns.barplot(
            x=[token for token, count in top_10_message_detail_tokens],
            y=[count for token, count in top_10_message_detail_tokens],
        )
        plt.title("Top 10 Tokens in Message Detail")
        plt.xlabel("Tokens")
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig("analysis/text_occurence.jpg", dpi=600)

    def analyze_label_cramers(self, data: pd.DataFrame):
        def cramers_v(confusion_matrix):
            """Calculate Cramér's V statistic for a given chi-squared contingency table"""
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

        # Creating cross-tabulations and calculating Cramér's V for each pair
        label_columns = ["level", "system", "person", "reason", "reason_detail"]
        cramers_v_results = {}
        cramers_v_results = {}
        label_pairs = [
            (label1, label2)
            for i, label1 in enumerate(label_columns)
            for label2 in label_columns[i + 1 :]
        ]

        for label1, label2 in label_pairs:
            confusion_matrix = pd.crosstab(data[label1], data[label2])
            cramers_v_results[f"{label1} & {label2}"] = cramers_v(confusion_matrix)

        print(cramers_v_results)

        # Converting the Cramér's V results to a DataFrame for visualization
        cramers_v_df = pd.DataFrame.from_dict(
            cramers_v_results, orient="index", columns=["Cramérs V"]
        )

        # Plotting the results
        plt.figure(figsize=(15, 9))
        sns.barplot(
            x=cramers_v_df.index, y=cramers_v_df["Cramérs V"], hue=cramers_v_df.index
        )
        plt.title("Cramér's V Statistic Between Hostname and Labels")
        plt.xlabel("Label Columns")
        plt.ylabel("Cramér's V")
        plt.ylim(0, 1)  # Cramér's V ranges from 0 to 1

        plt.savefig("analysis/label_cramers.jpg", dpi=600)

    def analyze_cramers(self, data: pd.DataFrame):
        def cramers_v(confusion_matrix):
            """Calculate Cramér's V statistic for a given chi-squared contingency table"""
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

        # Creating cross-tabulations and calculating Cramér's V for each pair
        cramers_v_results = {}

        for label in ["level", "system", "person", "reason", "reason_detail"]:
            confusion_matrix = pd.crosstab(data["hostname"], data[label])
            cramers_v_results[label] = cramers_v(confusion_matrix)

        cramers_v_results

        # Converting the Cramér's V results to a DataFrame for visualization
        cramers_v_df = pd.DataFrame.from_dict(
            cramers_v_results, orient="index", columns=["Cramérs V"]
        )

        # Plotting the results
        plt.figure(figsize=(15, 9))
        sns.barplot(
            x=cramers_v_df.index, y=cramers_v_df["Cramérs V"], hue=cramers_v_df.index
        )
        plt.title("Cramér's V Statistic Between Hostname and Labels")
        plt.xlabel("Label Columns")
        plt.ylabel("Cramér's V")
        plt.ylim(0, 1)  # Cramér's V ranges from 0 to 1

        plt.savefig("analysis/cramer.jpg", dpi=600)

    def analyze(self, file_path: str):
        raw_data_preprocessor: RawDataPreprocessor = RawDataPreprocessor(file_path)
        processed_df: pd.DataFrame = raw_data_preprocessor.get_processed_dataframe()

        self.analyze_text_length(processed_df)
        self.analyze_label_distribution(processed_df)
        # self.analyze_host_name(processed_df)
        self.analyze_token_occurance(processed_df)
        self.analyze_label_cramers(processed_df)
        self.analyze_cramers(processed_df)

    def analyze_post(self, file_path: str):
        post_raw_data_preprocessor: PostRawDataPreprocessor = PostRawDataPreprocessor(
            file_path
        )
        processed_df: pd.DataFrame = (
            post_raw_data_preprocessor.get_processed_dataframe()
        )

        self.analyze_text_length(processed_df)
        self.analyze_label_distribution(processed_df)
        self.analyze_host_name(processed_df)
        self.analyze_token_occurance(processed_df)
        self.analyze_label_cramers(processed_df)
        self.analyze_cramers(processed_df)


def analyze():
    analyzer: ExploratoryDataAnalyzer = ExploratoryDataAnalyzer()
    analyzer.analyze(GROUND_TRUTH_RAW_DATA)
    # analyzer.analyze_post(POST_GROUND_TRUTH_RAW_DATA)


def process_verified_df():
    post_raw_data_preprocessor: PostRawDataPreprocessor = PostRawDataPreprocessor(
        POST_GROUND_TRUTH_RAW_DATA
    )
    processed_df: pd.DataFrame = post_raw_data_preprocessor.get_processed_dataframe()

    # level data
    [level_features_df, level_label_df] = LevelDatasetPreprocessor.process(processed_df)
    [
        _,
        reason_detail_label_df,
    ] = ReasonDetailDatasetPreprocessor.process(processed_df)
    [_, system_label_df] = SystemDatasetPreprocessor.process(processed_df)
    [_, reason_label_df] = ReasonDatasetPreprocessor.process(processed_df)
    [_, person_label_df] = PersonDatasetPreprocessor.process(processed_df)

    # Split into train and test
    level_features_df.to_csv("clean_data/post_level_features.csv")
    level_label_df.to_csv("clean_data/post_level_label.csv")
    reason_detail_label_df.to_csv("clean_data/post_reason_detail_label.csv")
    system_label_df.to_csv("clean_data/post_system_label.csv")
    reason_label_df.to_csv("clean_data/post_reason_label.csv")
    person_label_df.to_csv("clean_data/post_person_label.csv")


if __name__ == "__main__":
    # main()
    process_verified_df()
    # analyze()
