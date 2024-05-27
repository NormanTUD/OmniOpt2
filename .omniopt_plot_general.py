# DESCRIPTION: Plot general job info
# EXPECTED FILES: pd.csv

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting tool for analyzing trial data.')
    parser.add_argument('--min', type=float, help='Minimum value for result filtering')
    parser.add_argument('--max', type=float, help='Maximum value for result filtering')
    parser.add_argument('--save_to_file', nargs='?', const='plot', type=str, help='Path to save the plot(s)')
    parser.add_argument('--exclude_params', action='append', nargs='+', help="Params to be ignored", default=[])
    parser.add_argument('--run_dir', type=str, help='Path to a CSV file', required=True)
    parser.add_argument('--result_column', type=str, help='Name of the result column', default="result")
    parser.add_argument('--delete_temp', help='Delete temp files', action='store_true', default=False)
    parser.add_argument('--darkmode', help='Enable darktheme', action='store_true', default=False)
    parser.add_argument('--print_to_command_line', help='Print plot to command line', action='store_true', default=False)
    parser.add_argument('--single', help='Print plot to command line', action='store_true', default=False)
    parser.add_argument('--bubblesize', type=int, help='Size of the bubbles', default=7)
    parser.add_argument('--merge_with_previous_runs', action='append', nargs='+', help="Run-Dirs to be merged with", default=[])
    parser.add_argument('--plot_type', action='append', nargs='+', help="Params to be ignored", default=[])
    parser.add_argument('--bins', type=int, help='Number of bins for distribution of results', default=10)
    parser.add_argument('--alpha', type=float, help='Transparency of plot bars', default=0.5)
    parser.add_argument('--no_legend', help='Disables legend (useless here)', action='store_true', default=False)
    parser.add_argument('--debug', help='Enable debug', action='store_true', default=False)
    return parser.parse_args()

def filter_data(dataframe, min_value=None, max_value=None):
    if min_value is not None:
        dataframe = dataframe[dataframe['result'] >= min_value]
    if max_value is not None:
        dataframe = dataframe[dataframe['result'] <= max_value]
    return dataframe

def plot_graph(dataframe, save_to_file=None):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    sns.boxplot(x='generation_method', y='result', data=dataframe)
    plt.title('Results by Generation Method')
    plt.xlabel('Generation Method')
    plt.ylabel('Result')

    plt.subplot(2, 2, 2)
    sns.countplot(x='trial_status', data=dataframe)
    plt.title('Distribution of Trial Status')
    plt.xlabel('Trial Status')
    plt.ylabel('Count')

    plt.subplot(2, 2, 3)
    exclude_columns = ['trial_index', 'arm_name', 'trial_status', 'generation_method']
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
    correlation_matrix = dataframe[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
    plt.title('Correlation Matrix')

    plt.subplot(2, 2, 4)
    histogram = sns.histplot(data=dataframe, x='result', hue='generation_method', multiple="stack", kde=True, bins=args.bins)
    for patch in histogram.patches:
        patch.set_alpha(args.alpha)
    plt.title('Distribution of Results by Generation Method')
    plt.xlabel('Result')
    plt.ylabel('Frequency')

    plt.tight_layout()

    if save_to_file:
        plt.savefig(save_to_file)
    else:
        plt.show()

def update_graph():
    try:
        dataframe = pd.read_csv(args.run_dir + "/pd.csv")

        if args.min is not None or args.max is not None:
            dataframe = filter_data(dataframe, args.min, args.max)

        if dataframe.empty:
            logging.warning("DataFrame is empty after filtering.")
            return

        plot_graph(dataframe, args.save_to_file)

    except FileNotFoundError:
        logging.error("File not found: %s", args.run_dir + "/pd.csv")
    except Exception as exception:
        logging.error("An unexpected error occurred: %s", str(exception))

if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()

    if not os.path.exists(args.run_dir):
        logging.error("Specified --run_dir does not exist")
        sys.exit(1)

    update_graph()