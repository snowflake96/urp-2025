"""
Visualization utilities for aneurysm analysis results.
Provides functions for visualizing meshes, stress fields, and predictions.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


class AneurysmVisualizer:
    """Visualization tools for aneurysm analysis."""
    
    def __init__(self, theme: str = 'document'):
        """
        Initialize visualizer.
        
        Args:
            theme: PyVista theme ('document', 'dark', 'paraview')
        """
        pv.set_plot_theme(theme)
        self.plotter = None
        
    def visualize_mesh_with_stress(self, mesh: Union[trimesh.Trimesh, pv.PolyData],
                                  stress_values: np.ndarray,
                                  stress_type: str = 'Von Mises',
                                  save_path: Optional[str] = None,
                                  show: bool = True) -> pv.Plotter:
        """
        Visualize mesh with stress field overlay.
        
        Args:
            mesh: Aneurysm mesh
            stress_values: Stress values at each vertex
            stress_type: Type of stress being visualized
            save_path: Optional path to save screenshot
            show: Whether to display the plot
            
        Returns:
            PyVista plotter object
        """
        # Convert trimesh to pyvista if needed
        if isinstance(mesh, trimesh.Trimesh):
            faces = np.hstack([[3] + list(face) for face in mesh.faces])
            mesh = pv.PolyData(mesh.vertices, faces)
            
        # Add stress data to mesh
        mesh[stress_type] = stress_values
        
        # Create plotter
        self.plotter = pv.Plotter()
        self.plotter.add_mesh(mesh, scalars=stress_type, 
                             cmap='jet', 
                             smooth_shading=True,
                             scalar_bar_args={'title': f'{stress_type} Stress (Pa)'})
        
        # Add axes
        self.plotter.show_axes()
        
        # Save screenshot if requested
        if save_path:
            self.plotter.screenshot(save_path)
            
        # Show if requested
        if show:
            self.plotter.show()
            
        return self.plotter
        
    def compare_meshes(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh,
                      labels: Tuple[str, str] = ('Before', 'After'),
                      save_path: Optional[str] = None) -> pv.Plotter:
        """
        Compare two meshes side by side.
        
        Args:
            mesh1: First mesh
            mesh2: Second mesh
            labels: Labels for the meshes
            save_path: Optional path to save screenshot
            
        Returns:
            PyVista plotter object
        """
        plotter = pv.Plotter(shape=(1, 2))
        
        # First mesh
        plotter.subplot(0, 0)
        plotter.add_text(labels[0], font_size=12, position='upper_edge')
        faces1 = np.hstack([[3] + list(face) for face in mesh1.faces])
        pv_mesh1 = pv.PolyData(mesh1.vertices, faces1)
        plotter.add_mesh(pv_mesh1, color='lightblue', smooth_shading=True)
        plotter.show_axes()
        
        # Second mesh
        plotter.subplot(0, 1)
        plotter.add_text(labels[1], font_size=12, position='upper_edge')
        faces2 = np.hstack([[3] + list(face) for face in mesh2.faces])
        pv_mesh2 = pv.PolyData(mesh2.vertices, faces2)
        plotter.add_mesh(pv_mesh2, color='lightcoral', smooth_shading=True)
        plotter.show_axes()
        
        # Link cameras
        plotter.link_views()
        
        if save_path:
            plotter.screenshot(save_path)
            
        plotter.show()
        return plotter
        
    def plot_feature_distributions(self, features_df: pd.DataFrame,
                                 rupture_column: str = 'rupture_status',
                                 save_path: Optional[str] = None):
        """
        Plot distributions of features for ruptured vs unruptured aneurysms.
        
        Args:
            features_df: DataFrame with features and rupture status
            rupture_column: Column name for rupture status
            save_path: Optional path to save plot
        """
        # Select numerical features
        numerical_features = features_df.select_dtypes(include=[np.number]).columns
        numerical_features = [f for f in numerical_features if f != rupture_column]
        
        # Create subplots
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(numerical_features):
            ax = axes[i]
            
            # Plot distributions
            for status in features_df[rupture_column].unique():
                data = features_df[features_df[rupture_column] == status][feature]
                ax.hist(data, alpha=0.6, label=status, bins=20)
                
            ax.set_xlabel(feature)
            ax.set_ylabel('Count')
            ax.legend()
            ax.set_title(f'Distribution of {feature}')
            
        # Remove empty subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_stress_evolution(self, time_points: List[float],
                            stress_values: List[Dict[str, np.ndarray]],
                            save_path: Optional[str] = None):
        """
        Plot evolution of stress metrics over time.
        
        Args:
            time_points: List of time points
            stress_values: List of dictionaries with stress metrics
            save_path: Optional path to save plot
        """
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Max Von Mises Stress',
                                         'Mean Von Mises Stress',
                                         'Max Wall Shear Stress',
                                         '95th Percentile Stress'))
        
        # Extract metrics
        max_vm = [sv['max_von_mises'] for sv in stress_values]
        mean_vm = [sv['mean_von_mises'] for sv in stress_values]
        max_wss = [sv['max_wss'] for sv in stress_values]
        p95_stress = [sv['95th_percentile_von_mises'] for sv in stress_values]
        
        # Add traces
        fig.add_trace(go.Scatter(x=time_points, y=max_vm, mode='lines+markers',
                                name='Max Von Mises'), row=1, col=1)
        fig.add_trace(go.Scatter(x=time_points, y=mean_vm, mode='lines+markers',
                                name='Mean Von Mises'), row=1, col=2)
        fig.add_trace(go.Scatter(x=time_points, y=max_wss, mode='lines+markers',
                                name='Max WSS'), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_points, y=p95_stress, mode='lines+markers',
                                name='95th Percentile'), row=2, col=2)
        
        # Update layout
        fig.update_xaxes(title_text="Time (months)")
        fig.update_yaxes(title_text="Stress (Pa)")
        fig.update_layout(height=800, showlegend=False,
                         title_text="Stress Evolution Over Time")
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
        
    def create_3d_voxel_visualization(self, voxel_grid: np.ndarray,
                                    threshold: float = 0.5,
                                    save_path: Optional[str] = None):
        """
        Visualize 3D voxel grid (e.g., from GAN predictions).
        
        Args:
            voxel_grid: 3D numpy array
            threshold: Threshold for voxel visualization
            save_path: Optional path to save visualization
        """
        # Create point cloud from voxel grid
        indices = np.where(voxel_grid > threshold)
        points = np.column_stack(indices)
        
        # Create PyVista point cloud
        point_cloud = pv.PolyData(points)
        
        # Create glyphs (small cubes) at each point
        glyphs = point_cloud.glyph(geom=pv.Cube(x_length=0.9, y_length=0.9, z_length=0.9))
        
        # Visualize
        plotter = pv.Plotter()
        plotter.add_mesh(glyphs, color='red', opacity=0.8)
        plotter.show_axes()
        plotter.show_grid()
        
        if save_path:
            plotter.screenshot(save_path)
            
        plotter.show()
        return plotter
        
    def plot_prediction_uncertainty(self, predictions: np.ndarray,
                                  uncertainties: np.ndarray,
                                  ground_truth: Optional[np.ndarray] = None,
                                  save_path: Optional[str] = None):
        """
        Plot predictions with uncertainty bands.
        
        Args:
            predictions: Predicted values
            uncertainties: Uncertainty estimates
            ground_truth: Optional ground truth values
            save_path: Optional path to save plot
        """
        x = np.arange(len(predictions))
        
        plt.figure(figsize=(12, 6))
        
        # Plot predictions with uncertainty
        plt.plot(x, predictions, 'b-', label='Predictions', linewidth=2)
        plt.fill_between(x, 
                        predictions - 2*uncertainties,
                        predictions + 2*uncertainties,
                        alpha=0.3, color='blue', label='95% Confidence')
        
        # Plot ground truth if available
        if ground_truth is not None:
            plt.plot(x, ground_truth, 'r--', label='Ground Truth', linewidth=2)
            
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('Predictions with Uncertainty')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_feature_importance_plot(self, feature_names: List[str],
                                     importances: np.ndarray,
                                     save_path: Optional[str] = None):
        """
        Create feature importance plot for ML models.
        
        Args:
            feature_names: List of feature names
            importances: Feature importance values
            save_path: Optional path to save plot
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        plt.figure(figsize=(10, 8))
        plt.barh(np.array(feature_names)[indices], importances[indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_confusion_matrix_plot(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   labels: List[str] = ['Unruptured', 'Ruptured'],
                                   save_path: Optional[str] = None):
        """
        Create confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            save_path: Optional path to save plot
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_summary_report(results_dir: str, output_path: str):
    """
    Create a summary visualization report.
    
    Args:
        results_dir: Directory containing analysis results
        output_path: Path for output HTML report
    """
    from plotly.subplots import make_subplots
    import plotly.io as pio
    
    # Create HTML template
    html_template = """
    <html>
    <head>
        <title>Aneurysm Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            h2 { color: #666; }
            .section { margin-bottom: 40px; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Cerebral Aneurysm Analysis Report</h1>
        <div class="section">
            <h2>Summary Statistics</h2>
            {summary_stats}
        </div>
        <div class="section">
            <h2>Feature Distributions</h2>
            {feature_plots}
        </div>
        <div class="section">
            <h2>Model Performance</h2>
            {model_performance}
        </div>
        <div class="section">
            <h2>Stress Analysis Results</h2>
            {stress_results}
        </div>
    </body>
    </html>
    """
    
    # Generate content (placeholder)
    summary_stats = "<p>Analysis completed on X aneurysms</p>"
    feature_plots = "<p>Feature distribution plots</p>"
    model_performance = "<p>Model accuracy: XX%</p>"
    stress_results = "<p>Average max stress: XX Pa</p>"
    
    # Fill template
    html_content = html_template.format(
        summary_stats=summary_stats,
        feature_plots=feature_plots,
        model_performance=model_performance,
        stress_results=stress_results
    )
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(html_content)
        
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    visualizer = AneurysmVisualizer()
    print("Visualization module loaded successfully") 