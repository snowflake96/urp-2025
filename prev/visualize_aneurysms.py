from pathlib import Path
import pandas as pd

class AneurysmVisualizer:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = Path.home() / "urp" / "data" / "aneurysm_cropping"
        self.output_dir = Path(output_dir)
        self.summary_file = self.output_dir / "aneurysm_cropping_summary.csv"
        self.summary_df = None
        
        if self.summary_file.exists():
            self.summary_df = pd.read_csv(self.summary_file)
            print(f"Loaded summary with {len(self.summary_df)} aneurysms")
        else:
            print("Summary file not found!") 