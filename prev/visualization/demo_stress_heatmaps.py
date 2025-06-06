#!/usr/bin/env python3
"""
Demo Script for 3D Stress Heatmap Visualizations
- Display summary of generated visualizations
- Provide access instructions for interactive viewing
- Show key statistics from the analysis
"""

import json
import pandas as pd
from pathlib import Path
import webbrowser
import argparse

def main():
    """Demo script for 3D stress heatmap visualizations"""
    
    # Paths
    viz_dir = Path("/home/jiwoo/repo/urp-2025/visualization/stress_heatmaps")
    results_dir = Path("/home/jiwoo/urp/data/uan/parallel_stress_results")
    
    print("=" * 70)
    print("🔬 3D STRESS HEATMAP VISUALIZATION DEMO")
    print("=" * 70)
    
    # Check if visualizations exist
    if not viz_dir.exists():
        print("❌ Visualization directory not found!")
        print("Run: python visualization/stress_3d_heatmap.py --max-individual 15")
        return
    
    # Load analysis data
    features_file = results_dir / "comprehensive_biomechanical_features.csv"
    clinical_file = results_dir / "comprehensive_clinical_report.json"
    
    if features_file.exists() and clinical_file.exists():
        df = pd.read_csv(features_file)
        with open(clinical_file, 'r') as f:
            clinical = json.load(f)
        
        # Display key statistics
        print(f"\n📊 ANALYSIS SUMMARY")
        print(f"   Total cases analyzed: {len(df)}")
        print(f"   High-risk cases (>7.0): {len(df[df['rupture_risk_score'] > 7.0])}")
        print(f"   Mean risk score: {df['rupture_risk_score'].mean():.1f}/10")
        print(f"   Mean max stress: {df['max_stress'].mean()/1000:.1f} kPa")
        print(f"   Mean safety factor: {df['safety_factor'].mean():.2f}")
        
        # Regional breakdown
        print(f"\n🏥 REGIONAL RISK ANALYSIS")
        regional = df.groupby('region')['rupture_risk_score'].agg(['count', 'mean', 'max']).round(1)
        for region, stats in regional.iterrows():
            risk_emoji = "🔴" if stats['mean'] > 7 else "🟡" if stats['mean'] > 5 else "🟢"
            print(f"   {risk_emoji} {region}: {stats['count']} cases, Risk {stats['mean']:.1f}/10")
    
    # List generated visualizations
    html_files = list(viz_dir.glob("*.html"))
    
    print(f"\n🎨 GENERATED VISUALIZATIONS ({len(html_files)} files)")
    print(f"   📁 Location: {viz_dir}")
    
    # Categorize files
    individual_heatmaps = [f for f in html_files if "patient_" in f.name and "stress_heatmap_3d" in f.name]
    overview_files = [f for f in html_files if f.name in ["population_stress_overview.html", "high_risk_comparison.html"]]
    report_files = [f for f in html_files if "report" in f.name]
    
    print(f"\n   📈 Population Analysis:")
    for f in overview_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"      • {f.name} ({size_mb:.1f} MB)")
    
    print(f"\n   🔍 Individual 3D Heatmaps ({len(individual_heatmaps)} cases):")
    for f in sorted(individual_heatmaps):
        size_mb = f.stat().st_size / (1024*1024)
        case_info = f.stem.replace("_stress_heatmap_3d", "").replace("patient_", "Patient ")
        print(f"      • {case_info} ({size_mb:.1f} MB)")
    
    print(f"\n   📋 Reports:")
    for f in report_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"      • {f.name} ({size_mb:.1f} MB)")
    
    # Access instructions
    print(f"\n🌐 HOW TO VIEW VISUALIZATIONS")
    print(f"   1. Navigate to: {viz_dir}")
    print(f"   2. Open any .html file in your web browser")
    print(f"   3. Interactive features:")
    print(f"      • Rotate: Click and drag")
    print(f"      • Zoom: Mouse wheel or pinch")
    print(f"      • Pan: Shift + click and drag")
    print(f"      • Reset view: Double-click")
    
    # Recommended viewing order
    print(f"\n⭐ RECOMMENDED VIEWING ORDER")
    print(f"   1. 📋 comprehensive_visualization_report.html - Start here!")
    print(f"   2. 📊 population_stress_overview.html - Population analysis")
    print(f"   3. ⚖️ high_risk_comparison.html - Risk comparison")
    print(f"   4. 🔴 Individual patient heatmaps - Detailed 3D analysis")
    
    # Top high-risk cases
    if features_file.exists():
        top_risk = df.nlargest(5, 'rupture_risk_score')[['patient_id', 'region', 'rupture_risk_score', 'max_stress', 'safety_factor']]
        print(f"\n🚨 TOP 5 HIGH-RISK CASES")
        for _, case in top_risk.iterrows():
            patient_id = int(case['patient_id'])
            region = case['region']
            risk = case['rupture_risk_score']
            stress = case['max_stress'] / 1000
            safety = case['safety_factor']
            heatmap_file = viz_dir / f"patient_{patient_id:02d}_{region}_stress_heatmap_3d.html"
            
            status = "✅ Available" if heatmap_file.exists() else "❌ Not generated"
            print(f"   🔴 Patient {patient_id:02d} - {region}")
            print(f"      Risk: {risk:.1f}/10 | Stress: {stress:.1f} kPa | Safety: {safety:.2f}")
            print(f"      3D Heatmap: {status}")
    
    # File size summary
    total_size = sum(f.stat().st_size for f in html_files) / (1024*1024)
    print(f"\n💾 STORAGE SUMMARY")
    print(f"   Total size: {total_size:.1f} MB")
    print(f"   Average per visualization: {total_size/len(html_files):.1f} MB")
    
    print(f"\n🎯 KEY FEATURES OF 3D HEATMAPS")
    print(f"   • Real vessel geometry from STL files")
    print(f"   • Stress distribution color mapping")
    print(f"   • High-stress region highlighting")
    print(f"   • Interactive 3D navigation")
    print(f"   • Clinical risk information overlay")
    print(f"   • Patient-specific biomechanical data")
    
    print(f"\n" + "=" * 70)
    print(f"🎉 3D Stress Heatmap Demo Complete!")
    print(f"📁 Open any HTML file in your browser to explore interactive visualizations")
    print(f"=" * 70)

if __name__ == "__main__":
    main() 