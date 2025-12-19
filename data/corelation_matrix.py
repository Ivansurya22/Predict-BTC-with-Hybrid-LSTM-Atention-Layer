import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Configuration
DB_PATH = os.getenv("OHLCV_DB_PATH", "data/btc_ohlcv.db")
TIMEFRAME = "btc_1h"


def load_features_for_correlation():
    """Load technical indicators and market regimes features."""
    smc_tables = {
        "technical_indicators": f"smc_{TIMEFRAME}_technical_indicators",
        "market_regimes": f"smc_{TIMEFRAME}_market_regimes",
    }

    features_dict = {}

    with sqlite3.connect(DB_PATH) as conn:
        for key, table in smc_tables.items():
            query = f"SELECT * FROM {table} ORDER BY timestamp"
            df = pd.read_sql(query, conn)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)

            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            features_dict[key] = df

    # Combine all features
    combined_df = pd.concat(features_dict.values(), axis=1)

    # Remove any remaining NaN or inf values
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    combined_df = combined_df.fillna(0)

    return combined_df


def calculate_correlation_matrix(df):
    """Calculate correlation matrix."""
    correlation_matrix = df.corr()
    return correlation_matrix


def plot_correlation_heatmap(corr_matrix, title="Feature Correlation Matrix"):
    """Plot correlation heatmap."""
    plt.figure(figsize=(20, 16))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=False,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    return plt.gcf()


def plot_clustered_correlation(corr_matrix, title="Clustered Correlation Matrix"):
    """Plot correlation matrix with hierarchical clustering."""
    plt.figure(figsize=(20, 16))

    # Perform hierarchical clustering
    dissimilarity = 1 - abs(corr_matrix)
    hierarchy_linkage = hierarchy.linkage(
        squareform(dissimilarity), method='average'
    )

    # Get cluster order
    dendro = hierarchy.dendrogram(hierarchy_linkage, no_plot=True)
    cluster_order = dendro['leaves']

    # Reorder correlation matrix
    corr_matrix_clustered = corr_matrix.iloc[cluster_order, cluster_order]

    # Plot
    sns.heatmap(
        corr_matrix_clustered,
        annot=False,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Features (Clustered)', fontsize=12)
    plt.ylabel('Features (Clustered)', fontsize=12)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    return plt.gcf()


def find_highly_correlated_features(corr_matrix, threshold=0.8):
    """Find pairs of highly correlated features."""
    # Get upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation above threshold
    high_corr_pairs = []
    for column in upper_triangle.columns:
        correlated = upper_triangle[column][abs(upper_triangle[column]) > threshold]
        for idx, value in correlated.items():
            high_corr_pairs.append({
                'Feature_1': column,
                'Feature_2': idx,
                'Correlation': value
            })

    # Sort by absolute correlation
    high_corr_df = pd.DataFrame(high_corr_pairs)
    if not high_corr_df.empty:
        high_corr_df['Abs_Correlation'] = abs(high_corr_df['Correlation'])
        high_corr_df = high_corr_df.sort_values('Abs_Correlation', ascending=False)
        high_corr_df = high_corr_df.drop('Abs_Correlation', axis=1)

    return high_corr_df


def suggest_feature_removal(corr_matrix, threshold=0.8):
    """Suggest features to remove based on high correlation."""
    high_corr = find_highly_correlated_features(corr_matrix, threshold)

    if high_corr.empty:
        return []

    # Create a set of features to remove
    to_remove = set()
    processed_pairs = set()

    for _, row in high_corr.iterrows():
        feat1, feat2 = row['Feature_1'], row['Feature_2']
        pair = tuple(sorted([feat1, feat2]))

        if pair not in processed_pairs:
            processed_pairs.add(pair)

            # Keep the feature that has lower average correlation with others
            avg_corr_1 = abs(corr_matrix[feat1]).mean()
            avg_corr_2 = abs(corr_matrix[feat2]).mean()

            if avg_corr_1 > avg_corr_2:
                to_remove.add(feat1)
            else:
                to_remove.add(feat2)

    return sorted(list(to_remove))


def analyze_feature_groups(df):
    """Analyze correlation within feature groups."""
    feature_groups = {
        'EMA': [col for col in df.columns if col.startswith('ema_')],
        'MACD': [col for col in df.columns if col.startswith('macd')],
        'Bollinger Bands': [col for col in df.columns if col.startswith('bb_')],
        'ADX/DI': [col for col in df.columns if 'adx' in col or 'di' in col],
        'HMM Regime': [col for col in df.columns if 'hmm_regime' in col],
        'Trend': [col for col in df.columns if 'trend' in col],
        'Volume': [col for col in df.columns if 'volume' in col and 'OBVolume' not in col],
        'Other': ['rsi_14', 'atr_14', 'obv']
    }

    group_stats = []

    for group_name, features in feature_groups.items():
        if not features:
            continue

        # Filter features that exist in dataframe
        existing_features = [f for f in features if f in df.columns]

        if len(existing_features) < 2:
            continue

        # Calculate average correlation within group
        group_corr = df[existing_features].corr()

        # Get upper triangle values (excluding diagonal)
        mask = np.triu(np.ones_like(group_corr, dtype=bool), k=1)
        upper_triangle_values = group_corr.values[mask]

        if len(upper_triangle_values) > 0:
            group_stats.append({
                'Group': group_name,
                'Features': len(existing_features),
                'Avg_Correlation': np.mean(abs(upper_triangle_values)),
                'Max_Correlation': np.max(abs(upper_triangle_values)),
                'Min_Correlation': np.min(abs(upper_triangle_values))
            })

    return pd.DataFrame(group_stats)


def print_summary_statistics(corr_matrix):
    """Print summary statistics of correlation matrix."""
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_triangle = corr_matrix.values[mask]

    print(f"\n{'═' * 60}")
    print("📊 CORRELATION SUMMARY STATISTICS")
    print(f"{'═' * 60}")
    print(f"Total features: {len(corr_matrix)}")
    print(f"Total correlations: {len(upper_triangle)}")
    print(f"\nCorrelation Statistics:")
    print(f"  Mean (absolute): {np.mean(abs(upper_triangle)):.4f}")
    print(f"  Median (absolute): {np.median(abs(upper_triangle)):.4f}")
    print(f"  Std deviation: {np.std(upper_triangle):.4f}")
    print(f"  Min: {np.min(upper_triangle):.4f}")
    print(f"  Max: {np.max(upper_triangle):.4f}")

    # Count by correlation ranges
    ranges = [
        (0.0, 0.3, "Low"),
        (0.3, 0.5, "Moderate"),
        (0.5, 0.7, "Strong"),
        (0.7, 0.9, "Very Strong"),
        (0.9, 1.0, "Extremely High")
    ]

    print(f"\nCorrelation Distribution:")
    for low, high, label in ranges:
        count = np.sum((abs(upper_triangle) >= low) & (abs(upper_triangle) < high))
        percentage = (count / len(upper_triangle)) * 100
        print(f"  {label:15s} ({low:.1f}-{high:.1f}): {count:4d} ({percentage:5.1f}%)")


def main():
    """Main function to analyze feature correlations."""
    print(f"\n{'═' * 60}")
    print("🔍 FEATURE CORRELATION ANALYSIS")
    print("   (Technical Indicators + Market Regimes)")
    print(f"{'═' * 60}")
    print(f"📂 Database: {DB_PATH}")
    print(f"🎯 Timeframe: {TIMEFRAME}")

    # Load features
    print(f"\n⏳ Loading features...")
    df_features = load_features_for_correlation()
    print(f"✅ Loaded {len(df_features)} rows × {len(df_features.columns)} features")
    print(f"\nFeature categories:")
    print(f"  - Technical Indicators: {len([c for c in df_features.columns if any(x in c for x in ['ema', 'macd', 'rsi', 'bb', 'atr', 'adx', 'di', 'obv'])])}")
    print(f"  - Market Regimes: {len([c for c in df_features.columns if any(x in c for x in ['hmm', 'trend', 'volume', 'alignment'])])}")

    # Calculate correlation matrix
    print(f"\n⏳ Calculating correlation matrix...")
    corr_matrix = calculate_correlation_matrix(df_features)
    print(f"✅ Correlation matrix: {corr_matrix.shape}")

    # Print summary statistics
    print_summary_statistics(corr_matrix)

    # Analyze feature groups
    print(f"\n{'─' * 60}")
    print("📊 FEATURE GROUP ANALYSIS")
    print(f"{'─' * 60}")
    group_stats = analyze_feature_groups(df_features)
    print(group_stats.to_string(index=False))

    # Find highly correlated features
    print(f"\n{'─' * 60}")
    print("⚠️  HIGH CORRELATION PAIRS (|r| > 0.8)")
    print(f"{'─' * 60}")
    high_corr = find_highly_correlated_features(corr_matrix, threshold=0.8)

    if high_corr.empty:
        print("✅ No highly correlated feature pairs found!")
    else:
        print(f"Found {len(high_corr)} pairs:\n")
        print(high_corr.to_string(index=False))

        # Suggest features to remove
        print(f"\n{'─' * 60}")
        print("💡 SUGGESTED FEATURES TO REMOVE")
        print(f"{'─' * 60}")
        to_remove = suggest_feature_removal(corr_matrix, threshold=0.8)

        if to_remove:
            print(f"Consider removing {len(to_remove)} features to reduce multicollinearity:")
            for feat in to_remove:
                avg_corr = abs(corr_matrix[feat]).mean()
                print(f"  - {feat:30s} (avg |r| = {avg_corr:.3f})")
        else:
            print("✅ No features need to be removed!")

    # Plot correlation heatmap
    print(f"\n⏳ Generating correlation heatmap...")
    fig1 = plot_correlation_heatmap(corr_matrix)
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: correlation_heatmap.png")
    plt.close()

    # Plot clustered correlation
    print(f"⏳ Generating clustered correlation heatmap...")
    fig2 = plot_clustered_correlation(corr_matrix)
    plt.savefig('correlation_clustered.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: correlation_clustered.png")
    plt.close()

    # Save correlation matrix to CSV
    print(f"\n⏳ Saving correlation matrix to CSV...")
    corr_matrix.to_csv('correlation_matrix.csv')
    print("✅ Saved: correlation_matrix.csv")

    # Save high correlation pairs to CSV
    if not high_corr.empty:
        high_corr.to_csv('high_correlation_pairs.csv', index=False)
        print("✅ Saved: high_correlation_pairs.csv")

    print(f"\n{'═' * 60}")
    print("✅ ANALYSIS COMPLETED")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
