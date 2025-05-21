#!/usr/bin/env python3
import json
import os
import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np

def load_experiment_results(results_file: str) -> List[Dict[str, Any]]:
    """Load results from a JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def process_trial_data(result: Dict[str, Any]) -> Tuple[bool, float, List[Dict[str, Any]]]:
    """Process trial data from a result, returning first trial success, time and all trials data."""
    history = result['metadata'].get('llm_history', [])
    if not history:  # No trial data available
        return result['valid'], result['solve_time'], []
    
    # First trial data
    first_trial = history[0]
    first_trial_valid = not bool(first_trial.get('invalid_first'))
    first_trial_time = result['solve_time'] / len(history)  # Approximate time per trial
    
    # Process all trials
    trial_data = []
    for i, trial in enumerate(history):
        trial_data.append({
            'trial_number': i + 1,
            'valid': not bool(trial.get('invalid_first')),
            'solve_time': first_trial_time,  # Using approximated time per trial
            'path_length': len(trial['path']) if 'path' in trial else 0
        })
    
    return first_trial_valid, first_trial_time, trial_data

def convert_to_dataframe(results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert results to two DataFrames: one for overall results and one for trial-level data."""
    overall_data = []
    trial_level_data = []
    
    for result in results:
        maze_size = result['maze_id'][0] if isinstance(result['maze_id'], list) else None
        maze_number = result['maze_id'][1] if isinstance(result['maze_id'], list) else None
        
        # Process trial information
        first_trial_valid, first_trial_time, trial_data = process_trial_data(result)
        trials_taken = len(trial_data) if trial_data else 1
        
        # Add overall result data
        overall_data.append({
            'solver_name': result['solver_name'],
            'maze_size': maze_size,
            'maze_number': maze_number,
            'valid': result['valid'],
            'solve_time': result['solve_time'],
            'path_length': len(result['path']) if result['path'] else 0,
            'trials_taken': trials_taken,
            'first_trial_valid': first_trial_valid,
            'first_trial_time': first_trial_time
        })
        
        # Add trial-level data if available
        if trial_data:
            for trial in trial_data:
                trial_level_data.append({
                    'solver_name': result['solver_name'],
                    'maze_size': maze_size,
                    'maze_number': maze_number,
                    'trial_number': trial['trial_number'],
                    'valid': trial['valid'],
                    'solve_time': trial['solve_time'],
                    'path_length': trial['path_length']
                })
    
    return pd.DataFrame(overall_data), pd.DataFrame(trial_level_data)

def plot_solve_times_by_size(df: pd.DataFrame, trial_df: pd.DataFrame, output_dir: str):
    """Plot solve times grouped by maze size for each solver, comparing first trial vs all trials."""
    # Only include solvers that have trial data
    solvers_with_trials = trial_df['solver_name'].unique()
    
    if len(solvers_with_trials) > 0:
        plt.figure(figsize=(15, 6))
        
        # First trial times
        plt.subplot(1, 2, 1)
        solver_data = df[df['solver_name'].isin(solvers_with_trials)]
        sns.boxplot(data=solver_data, x='maze_size', y='first_trial_time', hue='solver_name')
        plt.title('First Trial Solve Times by Maze Size')
        plt.xlabel('Maze Size')
        plt.ylabel('Solve Time (seconds)')
        
        # All trials times
        plt.subplot(1, 2, 2)
        sns.boxplot(data=trial_df, x='maze_size', y='solve_time', hue='solver_name')
        plt.title('All Trials Solve Times by Maze Size')
        plt.xlabel('Maze Size')
        plt.ylabel('Solve Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'solve_times_comparison.pdf'))
        plt.close()
    
    # Original solve times plot for all solvers
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='maze_size', y='solve_time', hue='solver_name')
    plt.title('Overall Solve Times by Maze Size')
    plt.xlabel('Maze Size')
    plt.ylabel('Solve Time (seconds)')
    plt.savefig(os.path.join(output_dir, 'solve_times_by_size.pdf'))
    plt.close()

def plot_success_rate_by_size(df: pd.DataFrame, trial_df: pd.DataFrame, output_dir: str):
    """For each solver with >1 trial, plot success rate by maze size, with a line for each trial N (accumulated success rate by or before trial N)."""
    # Find solvers with max(trials_taken) > 1
    solver_trial_counts = df.groupby('solver_name')['trials_taken'].max()
    solvers_with_multiple_trials = solver_trial_counts[solver_trial_counts > 1].index.tolist()

    if len(solvers_with_multiple_trials) == 0:
        print("No solvers attempted more than one trial. No plot generated.")
        return

    linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1))]  # cycle through some styles
    markers = ['s', 'D', 'v', '^', 'P', 'X', '*']

    for solver in solvers_with_multiple_trials:
        solver_trials = trial_df[trial_df['solver_name'] == solver]
        if solver_trials.empty:
            continue
        max_trial = solver_trials['trial_number'].max()
        maze_sizes = sorted(solver_trials['maze_size'].unique())

        plt.figure(figsize=(12, 6))
        # Plot the final success rate line for this solver
        filtered_df = df[df['solver_name'] == solver]
        success_rates = filtered_df.groupby(['maze_size'])['valid'].mean().reset_index()
        
        # Filter out maze sizes where previous size had 0 accuracy
        valid_sizes = []
        for i, size in enumerate(success_rates['maze_size']):
            if i == 0 or success_rates.iloc[i-1]['valid'] > 0:
                valid_sizes.append(size)
        success_rates = success_rates[success_rates['maze_size'].isin(valid_sizes)]
        
        plt.plot(valid_sizes, success_rates['valid'], label=f"{solver} (final)", marker='o', color='black', linewidth=2)

        # Now plot accumulated success rate at each trial N for this solver
        for n in range(1, max_trial + 1):
            by_maze = solver_trials[solver_trials['trial_number'] <= n].groupby(['maze_size', 'maze_number'])['valid'].max().reset_index()
            by_size = by_maze.groupby('maze_size')['valid'].mean().reset_index()
            # Filter to same valid sizes as final success rates
            by_size = by_size[by_size['maze_size'].isin(valid_sizes)]
            label = f"≤trial {n}"
            style = linestyles[(n-1) % len(linestyles)]
            marker = markers[(n-1) % len(markers)]
            plt.plot(by_size['maze_size'], by_size['valid'], linestyle=style, marker=marker, label=label, alpha=0.7)

        plt.title(f'Success Rate by Maze Size ({solver}, by Trial N)')
        plt.xlabel('Maze Size')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Force integer ticks on x-axis
        plt.legend(title='Trial', loc='upper right')
        plt.tight_layout()
        safe_solver_name = solver.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f'success_rate_by_size_{safe_solver_name}_multiple_trials.pdf'))
        plt.close()

    # Plot solve rate by maze size for all solvers (each as a separate line)
    all_success_rates = df.groupby(['solver_name', 'maze_size'])['valid'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=all_success_rates, x='maze_size', y='valid', hue='solver_name', marker='o')
    plt.title('Success Rate by Maze Size (All Solvers)')
    plt.xlabel('Maze Size')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.legend(title='Solver', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_by_size_all_solvers.pdf'))
    plt.close()

def generate_summary_stats(df: pd.DataFrame, trial_df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for each solver."""
    stats = []
    for solver in df['solver_name'].unique():
        solver_data = df[df['solver_name'] == solver]
        solver_trials = trial_df[trial_df['solver_name'] == solver] if not trial_df.empty else pd.DataFrame()
        
        stats_dict = {
            'solver_name': solver,
            'total_mazes': len(solver_data),
            'success_rate': solver_data['valid'].mean() * 100,
            'avg_solve_time': solver_data['solve_time'].mean(),
            'median_solve_time': solver_data['solve_time'].median(),
            'avg_path_length': solver_data[solver_data['valid']]['path_length'].mean()
        }
        
        # Add trial-specific stats if available
        if not solver_trials.empty:
            stats_dict.update({
                'first_trial_success_rate': solver_data['first_trial_valid'].mean() * 100,
                'avg_trials_needed': solver_data['trials_taken'].mean(),
                'max_trials_needed': solver_data['trials_taken'].max()
            })
        
        stats.append(stats_dict)
    
    return pd.DataFrame(stats)

@click.command()
@click.argument('experiment_dirs', nargs=-1, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output-dir', type=click.Path(file_okay=False, dir_okay=True), default='comparison_plots',
              help='Directory to save the comparison plots')
def main(experiment_dirs, output_dir):
    """Compare results from multiple experiment directories.
    
    EXPERIMENT_DIRS: One or more directories containing solver_results.json files
    """
    if not experiment_dirs:
        click.echo("Please provide at least one experiment directory")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    all_results = []
    for exp_dir in experiment_dirs:
        results_file = os.path.join(exp_dir, 'solver_results.json')
        if not os.path.exists(results_file):
            click.echo(f"Warning: No solver_results.json found in {exp_dir}")
            continue
            
        results = load_experiment_results(results_file)
        # Add experiment directory name to solver names for disambiguation
        exp_name = os.path.basename(os.path.normpath(exp_dir))
        for result in results:
            result['solver_name'] = f"{result['solver_name']}-{exp_name}"
        all_results.extend(results)
    
    if not all_results:
        click.echo("No results found in any of the provided directories")
        return
    
    # Convert to DataFrame
    df, trial_df = convert_to_dataframe(all_results)
    
    # Generate plots
    click.echo("Generating comparison plots...")
    plot_solve_times_by_size(df, trial_df, output_dir)
    plot_success_rate_by_size(df, trial_df, output_dir)
    
    # Generate and save summary statistics
    stats_df = generate_summary_stats(df, trial_df)
    stats_file = os.path.join(output_dir, 'summary_stats.csv')
    stats_df.to_csv(stats_file, index=False)
    
    click.echo(f"\nSummary statistics saved to {stats_file}")
    click.echo("\nSummary of results:")
    click.echo(stats_df.to_string(index=False))
    click.echo(f"\nPlots have been saved to {output_dir}")

if __name__ == '__main__':
    main() 