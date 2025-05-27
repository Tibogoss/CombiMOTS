import pandas as pd
import argparse
import numpy as np

def filter_and_sample_dual_activity(input_path, output_path, n_samples=1000, random_seed=42):
    """
    Filter CSV for molecules with both targets' activities above threshold,
    then sort by high docking scores with minimal differences between them.
    """
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    df = df.dropna(subset=['dockingscore1', 'dockingscore2'])
    print(f"Dataset size after removing missing docking scores: {len(df)}")
    
    # Restore original docking score scale (*20)
    df['dockingscore1'] = -df['dockingscore1'] * 20
    df['dockingscore2'] = -df['dockingscore2'] * 20
    
    # Filter for molecules with both activities above threshold
    df_filtered = df[
        (df['activity1'] >= 0.5) & 
        (df['activity2'] >= 0.5)
    ]
    
    print(f"\nFiltering results:")
    print(f"Original dataset size: {len(df)}")
    print(f"Molecules with both activities >= 0.5: {len(df_filtered)}")

    # Filter for molecules with only one activity above threshold
    t1_filtered = df[df['activity1'] >= 0.5]
    t2_filtered = df[df['activity2'] >= 0.5]

    print(f"Molecules with target1 activity >= 0.5: {len(t1_filtered)}")
    print(f"Molecules with target2 activity >= 0.5: {len(t2_filtered)}")

    # Check if any dual actives are found    
    if len(df_filtered) == 0:
        print(f"No molecules found with both activities >= 0.5")
        return None
    
    df_filtered['avg_dockingscore'] = df_filtered[['dockingscore1', 'dockingscore2']].mean(axis=1)
    df_filtered['dockingscore_diff'] = np.abs(df_filtered['dockingscore1'] - df_filtered['dockingscore2'])
    
    df_filtered = df_filtered.sort_values(
        ['avg_dockingscore', 'dockingscore_diff'], 
        ascending=[False, True]
    )
    
    if n_samples > len(df_filtered):
        print(f"Warning: Requested sample size ({n_samples}) is larger than filtered dataset size ({len(df_filtered)})")
        print("Using entire filtered dataset instead")
        sampled_df = df_filtered.copy()
    else:
        # Take top n_samples after sorting
        sampled_df = df_filtered.head(n_samples)
    
    sampled_df = sampled_df[['node_id', 'smiles', 'activity1', 'activity2', 'dockingscore1', 'dockingscore2', 
                            'avg_dockingscore', 'dockingscore_diff']]
    
    sampled_df.to_csv(output_path, index=False)
    
    print(f"Sampled dataset size: {len(sampled_df)}")
    print(f"Saved sampled data to: {output_path}")
    
    # statistics 
    print("\nSampled data statistics:")
    print(f"activity1:")
    print(f"  Mean: {sampled_df['activity1'].mean():.3f}")
    print(f"  Min:  {sampled_df['activity1'].min():.3f}")
    print(f"  Max:  {sampled_df['activity1'].max():.3f}")
    print(f"activity2:")
    print(f"  Mean: {sampled_df['activity2'].mean():.3f}")
    print(f"  Min:  {sampled_df['activity2'].min():.3f}")
    print(f"  Max:  {sampled_df['activity2'].max():.3f}")
    print(f"Target 1 Docking Score:")
    print(f"  Mean: {sampled_df['dockingscore1'].mean():.3f}")
    print(f"  Min:  {sampled_df['dockingscore1'].min():.3f}")
    print(f"  Max:  {sampled_df['dockingscore1'].max():.3f}")
    print(f"Target 2 Docking Score:")
    print(f"  Mean: {sampled_df['dockingscore2'].mean():.3f}")
    print(f"  Min:  {sampled_df['dockingscore2'].min():.3f}")
    print(f"  Max:  {sampled_df['dockingscore2'].max():.3f}")
    print(f"Overall Average Docking Score:")
    print(f"  Mean: {sampled_df['avg_dockingscore'].mean():.3f}")
    print(f"  Min:  {sampled_df['avg_dockingscore'].min():.3f}")
    print(f"  Max:  {sampled_df['avg_dockingscore'].max():.3f}")
    print(f"Docking Score Difference:")
    print(f"  Mean: {sampled_df['dockingscore_diff'].mean():.3f}")
    print(f"  Min:  {sampled_df['dockingscore_diff'].min():.3f}")
    print(f"  Max:  {sampled_df['dockingscore_diff'].max():.3f}")
    
    return sampled_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter for dual activity and sample entries from CSV file')
    parser.add_argument('input_path', type=str, help='Path to input CSV file')
    parser.add_argument('output_path', type=str, help='Path for output CSV file')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples to take')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    sampled_df = filter_and_sample_dual_activity(
        args.input_path,
        args.output_path,
        args.n_samples,
        args.random_seed
    )