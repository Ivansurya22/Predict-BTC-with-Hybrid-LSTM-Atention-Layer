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


def load_all_features_for_correlation():
    """Load ONLY Technical Indicators + Market Regimes (exclude SMC features)."""
    # Only load features used in the model
    model_tables = {
        "technical_indicators": f"smc_{TIMEFRAME}_technical_indicators",
        "market_regimes": f"smc_{TIMEFRAME}_market_regimes",
    }

    features_dict = {}

    with sqlite3.connect(DB_PATH) as conn:
        for key, table in model_tables.items():
            try:
                query = f"SELECT * FROM {table} ORDER BY timestamp"
                df = pd.read_sql(query, conn)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df.set_index("timestamp", inplace=True)

                # Convert all columns to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                features_dict[key] = df
                print(f"  ✓ Loaded {key}: {len(df.columns)} features")
            except Exception as e:
                print(f"  ✗ Failed to load {key}: {e}")

    # Combine all features
    if not features_dict:
        raise ValueError("No features loaded!")

    combined_df = pd.concat(features_dict.values(), axis=1)

    # Remove any remaining NaN or inf values
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    combined_df = combined_df.fillna(0)

    return combined_df, features_dict


def load_features_by_category():
    """Load features separated by category for detailed analysis."""
    categories = {
        "Technical Indicators": f"smc_{TIMEFRAME}_technical_indicators",
        "Market Regimes": f"smc_{TIMEFRAME}_market_regimes",
    }

    features_by_category = {}

    with sqlite3.connect(DB_PATH) as conn:
        for category, table in categories.items():
            try:
                query = f"SELECT * FROM {table} ORDER BY timestamp"
                df = pd.read_sql(query, conn)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df.set_index("timestamp", inplace=True)

                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
                features_by_category[category] = df
            except Exception as e:
                print(f"  Warning: {category} not loaded: {e}")

    return features_by_category


def calculate_correlation_matrix(df):
    """Calculate correlation matrix."""
    correlation_matrix = df.corr()
    return correlation_matrix


def plot_correlation_heatmap(corr_matrix, title="Feature Correlation Matrix", filename="correlation_heatmap.png"):
    """Plot correlation heatmap."""
    plt.figure(figsize=(24, 20))

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

    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


def plot_clustered_correlation(corr_matrix, title="Clustered Correlation Matrix", filename="correlation_clustered.png"):
    """Plot correlation matrix with hierarchical clustering."""
    plt.figure(figsize=(24, 20))

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

    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Features (Clustered)', fontsize=14)
    plt.ylabel('Features (Clustered)', fontsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


def find_highly_correlated_features(corr_matrix, threshold=0.8):
    """Find pairs of highly correlated features."""
    # Get upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation above threshold
    high_corr_pairs = []
    for column in upper_triangle.columns:
        # Get correlations for this column
        col_corr = upper_triangle[column]
        # Filter by threshold using boolean indexing
        mask = col_corr.abs() > threshold
        correlated = col_corr[mask]

        for idx, value in correlated.items():
            high_corr_pairs.append({
                'Feature_1': column,
                'Feature_2': idx,
                'Correlation': float(value)  # Convert to float explicitly
            })

    # Create DataFrame
    if len(high_corr_pairs) == 0:
        return pd.DataFrame(columns=['Feature_1', 'Feature_2', 'Correlation'])

    high_corr_df = pd.DataFrame(high_corr_pairs)

    # Add absolute correlation column for sorting
    high_corr_df['Abs_Correlation'] = high_corr_df['Correlation'].abs()

    # Sort by absolute correlation
    high_corr_df = high_corr_df.sort_values(
        'Abs_Correlation', ascending=False).reset_index(drop=True)

    # Drop the helper column
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
    """Analyze correlation within feature groups (Model features only)."""
    feature_groups = {
        # Technical Indicators (16 features)
        'EMA': [col for col in df.columns if col.startswith('ema_')],
        'MACD': [col for col in df.columns if col.startswith('macd')],
        'Bollinger Bands': [col for col in df.columns if col.startswith('bb_')],
        'Momentum': ['rsi_14', 'atr_14', 'adx_14'],
        'Volume (OBV)': ['obv'],

        # Market Regimes (6 features)
        'HMM Regime': [col for col in df.columns if 'hmm_regime' in col],
        'Trend Regime': [col for col in df.columns if 'trend' in col or 'ema_alignment' in col],
        'Volume Regime': ['volume_percentile', 'volume_trend'],
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

    return pd.DataFrame(group_stats).sort_values('Avg_Correlation', ascending=False)


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
        count = np.sum((abs(upper_triangle) >= low) &
                       (abs(upper_triangle) < high))
        percentage = (count / len(upper_triangle)) * 100
        print(
            f"  {label:15s} ({low:.1f}-{high:.1f}): {count:4d} ({percentage:5.1f}%)")


def print_feature_breakdown(features_dict):
    """Print breakdown of loaded features."""
    print(f"\n{'─' * 60}")
    print("📋 FEATURE BREAKDOWN")
    print(f"{'─' * 60}")

    total_features = 0
    for category, df in features_dict.items():
        feature_count = len(df.columns)
        total_features += feature_count
        print(f"{category:25s}: {feature_count:3d} features")

        # Show feature names for each category
        feature_names = ', '.join(df.columns[:5])
        if len(df.columns) > 5:
            feature_names += f", ... (+{len(df.columns)-5} more)"
        print(f"  → {feature_names}")

    print(f"{'─' * 60}")
    print(f"{'TOTAL':25s}: {total_features:3d} features")


def analyze_category_correlations(features_by_category):
    """Analyze correlations within each category separately."""
    print(f"\n{'═' * 60}")
    print("📊 CATEGORY-SPECIFIC CORRELATION ANALYSIS")
    print(f"{'═' * 60}")

    for category, df in features_by_category.items():
        print(f"\n{category}")
        print(f"{'─' * 60}")

        if len(df.columns) < 2:
            print("  ⚠️  Not enough features for correlation analysis")
            continue

        corr_matrix = df.corr()

        # Get upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle = corr_matrix.values[mask]

        print(f"  Features: {len(df.columns)}")
        print(f"  Avg |correlation|: {np.mean(abs(upper_triangle)):.4f}")
        print(f"  Max |correlation|: {np.max(abs(upper_triangle)):.4f}")

        # Find highly correlated pairs in this category
        high_corr = find_highly_correlated_features(corr_matrix, threshold=0.8)
        if not high_corr.empty:
            print(f"  ⚠️  High correlation pairs: {len(high_corr)}")
            for _, row in high_corr.head(3).iterrows():
                print(
                    f"     • {row['Feature_1']} ↔ {row['Feature_2']}: {row['Correlation']:.3f}")
        else:
            print(f"  ✅ No high correlation pairs")


def main():
    """Main function to analyze feature correlations."""
    print(f"\n{'═' * 60}")
    print("🔍 MODEL FEATURE CORRELATION ANALYSIS")
    print("   (Technical Indicators + Market Regimes Only)")
    print(f"{'═' * 60}")
    print(f"📂 Database: {DB_PATH}")
    print(f"🎯 Timeframe: {TIMEFRAME}")
    print(f"ℹ️  Note: SMC features excluded (not used in model)")

    # Load MODEL features only
    print(f"\n⏳ Loading model features...")
    df_model_features, features_dict = load_all_features_for_correlation()
    print(
        f"\n✅ Loaded {len(df_model_features)} rows × {len(df_model_features.columns)} features")

    # Print feature breakdown
    print_feature_breakdown(features_dict)

    # Calculate correlation matrix
    print(f"\n⏳ Calculating correlation matrix...")
    corr_matrix = calculate_correlation_matrix(df_model_features)
    print(f"✅ Correlation matrix: {corr_matrix.shape}")

    # Print summary statistics
    print_summary_statistics(corr_matrix)

    # Analyze feature groups
    print(f"\n{'─' * 60}")
    print("📊 FEATURE GROUP ANALYSIS")
    print(f"{'─' * 60}")
    group_stats = analyze_feature_groups(df_model_features)
    if not group_stats.empty:
        print(group_stats.to_string(index=False))
    else:
        print("No feature groups found")

    # Find highly correlated features
    print(f"\n{'─' * 60}")
    print("⚠️  HIGH CORRELATION PAIRS (|r| > 0.8)")
    print(f"{'─' * 60}")
    high_corr = find_highly_correlated_features(corr_matrix, threshold=0.8)

    if high_corr.empty:
        print("✅ No highly correlated feature pairs found!")
    else:
        print(f"Found {len(high_corr)} pairs:\n")
        # Show all pairs
        print(high_corr.to_string(index=False))

        # Suggest features to remove
        print(f"\n{'─' * 60}")
        print("💡 SUGGESTED FEATURES TO REMOVE")
        print(f"{'─' * 60}")
        to_remove = suggest_feature_removal(corr_matrix, threshold=0.8)

        if to_remove:
            print(
                f"Consider removing {len(to_remove)} features to reduce multicollinearity:")
            for feat in to_remove:
                avg_corr = abs(corr_matrix[feat]).mean()
                print(f"  - {feat:30s} (avg |r| = {avg_corr:.3f})")
        else:
            print("✅ No features need to be removed!")

    # Analyze by category
    print(f"\n⏳ Analyzing categories separately...")
    features_by_category = load_features_by_category()
    analyze_category_correlations(features_by_category)

    # Plot correlation heatmap
    print(f"\n⏳ Generating correlation heatmap...")
    plot_correlation_heatmap(
        corr_matrix,
        title=f"Model Features Correlation Matrix ({len(corr_matrix)} features)",
        filename='correlation_heatmap.png'
    )

    # Plot clustered correlation
    print(f"⏳ Generating clustered correlation heatmap...")
    plot_clustered_correlation(
        corr_matrix,
        title=f"Clustered Correlation Matrix ({len(corr_matrix)} features)",
        filename='correlation_clustered.png'
    )

    # Generate separate heatmaps for Technical Indicators and Market Regimes
    if 'Technical Indicators' in features_by_category:
        print(f"\n⏳ Generating Technical Indicators correlation heatmap...")
        corr_tech = features_by_category['Technical Indicators'].corr()
        plot_correlation_heatmap(
            corr_tech,
            title="Technical Indicators Correlation (16 features)",
            filename='correlation_technical_indicators.png'
        )

    if 'Market Regimes' in features_by_category:
        print(f"⏳ Generating Market Regimes correlation heatmap...")
        corr_regimes = features_by_category['Market Regimes'].corr()
        plot_correlation_heatmap(
            corr_regimes,
            title="Market Regimes Correlation (6 features)",
            filename='correlation_market_regimes.png'
        )

    # Save correlation matrix to CSV
    print(f"\n⏳ Saving correlation matrices to CSV...")
    corr_matrix.to_csv('correlation_matrix.csv')
    print("✅ Saved: correlation_matrix.csv")

    # Save high correlation pairs to CSV
    if not high_corr.empty:
        high_corr.to_csv('high_correlation_pairs.csv', index=False)
        print("✅ Saved: high_correlation_pairs.csv")

    # Save feature list with statistics
    feature_list = pd.DataFrame({
        'Feature': df_model_features.columns,
        'Mean': df_model_features.mean(),
        'Std': df_model_features.std(),
        'Min': df_model_features.min(),
        'Max': df_model_features.max()
    })
    feature_list.to_csv('feature_statistics.csv', index=False)
    print("✅ Saved: feature_statistics.csv")

    print(f"\n{'═' * 60}")
    print("✅ ANALYSIS COMPLETED")
    print(f"{'═' * 60}")
    print("\n📊 Generated files:")
    print("  Visualizations:")
    print("    • correlation_heatmap.png - All model features")
    print("    • correlation_clustered.png - Hierarchical clustering")
    print("    • correlation_technical_indicators.png - Technical only")
    print("    • correlation_market_regimes.png - Regimes only")
    print("  Data:")
    print("    • correlation_matrix.csv - Full correlation matrix")
    print("    • high_correlation_pairs.csv - Highly correlated pairs")
    print("    • feature_statistics.csv - Feature statistics")

    # Print final summary
    print(f"\n📈 Summary:")
    print(f"  Total features analyzed: {len(df_model_features.columns)}")
    print(f"    - Technical Indicators: 16")
    print(f"    - Market Regimes: 6")
    if not high_corr.empty:
        print(f"  High correlation pairs (|r| > 0.8): {len(high_corr)}")
        if to_remove:
            print(f"  Suggested features to remove: {len(to_remove)}")
    print()


if __name__ == "__main__":
    main()
