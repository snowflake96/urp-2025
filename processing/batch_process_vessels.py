#!/usr/bin/env python3
"""
Batch Processing Script for Vessel Pipeline

Process multiple patients through the comprehensive vessel processing pipeline.
"""

import argparse
import json
from pathlib import Path
import subprocess
import logging
from typing import List, Dict
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd


def setup_logger(log_file: str = "batch_processing.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_patient_files(data_dir: Path, pattern: str = "*_MRA*_seg.nii.gz") -> List[Path]:
    """Get list of patient files to process"""
    files = list(data_dir.glob(pattern))
    return sorted(files)


def process_single_patient(nifti_file: Path, output_base_dir: Path, 
                         aneurysm_json: Path, 
                         pipeline_script: Path = Path("processing/vessel_processing_pipeline.py"),
                         extra_args: List[str] = None) -> Dict:
    """
    Process a single patient file
    
    Returns:
    --------
    dict : Processing results with status, timing, etc.
    """
    patient_id = nifti_file.stem
    output_dir = output_base_dir / patient_id
    
    start_time = time.time()
    
    # Build command
    cmd = [
        "python", str(pipeline_script),
        str(nifti_file),
        str(output_dir),
        "--aneurysm-json", str(aneurysm_json)
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    # Run pipeline
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Load results if available
        results_file = output_dir / "pipeline_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                pipeline_results = json.load(f)
        else:
            pipeline_results = {}
        
        return {
            'patient_id': patient_id,
            'status': 'success',
            'duration': time.time() - start_time,
            'output_dir': str(output_dir),
            'pipeline_results': pipeline_results,
            'stdout': result.stdout[-500:],  # Last 500 chars
            'error': None
        }
        
    except subprocess.CalledProcessError as e:
        return {
            'patient_id': patient_id,
            'status': 'failed',
            'duration': time.time() - start_time,
            'output_dir': str(output_dir),
            'pipeline_results': {},
            'stdout': e.stdout[-500:] if e.stdout else '',
            'error': e.stderr[-1000:] if e.stderr else str(e)
        }
    except Exception as e:
        return {
            'patient_id': patient_id,
            'status': 'error',
            'duration': time.time() - start_time,
            'output_dir': str(output_dir),
            'pipeline_results': {},
            'stdout': '',
            'error': str(e)
        }


def batch_process(nifti_files: List[Path], output_base_dir: Path,
                 aneurysm_json: Path, max_workers: int = 4,
                 extra_args: List[str] = None) -> pd.DataFrame:
    """
    Process multiple patients in parallel
    
    Returns:
    --------
    pd.DataFrame : Summary of all processing results
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting batch processing of {len(nifti_files)} files")
    logger.info(f"Using {max_workers} parallel workers")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(
                process_single_patient,
                nifti_file,
                output_base_dir,
                aneurysm_json,
                extra_args=extra_args
            ): nifti_file
            for nifti_file in nifti_files
        }
        
        # Process completed jobs
        for future in as_completed(future_to_file):
            nifti_file = future_to_file[future]
            
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    logger.info(f"✓ {result['patient_id']} - completed in {result['duration']:.1f}s")
                else:
                    logger.error(f"✗ {result['patient_id']} - {result['status']}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"✗ {nifti_file.stem} - Exception: {e}")
                results.append({
                    'patient_id': nifti_file.stem,
                    'status': 'exception',
                    'duration': 0,
                    'error': str(e)
                })
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Add summary statistics
    logger.info("\n=== Batch Processing Summary ===")
    logger.info(f"Total processed: {len(df)}")
    logger.info(f"Successful: {len(df[df['status'] == 'success'])}")
    logger.info(f"Failed: {len(df[df['status'] != 'success'])}")
    logger.info(f"Total time: {df['duration'].sum():.1f}s")
    logger.info(f"Average time per patient: {df['duration'].mean():.1f}s")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Batch process vessel data through comprehensive pipeline"
    )
    parser.add_argument('data_dir', help='Directory containing NIfTI files')
    parser.add_argument('output_dir', help='Base output directory')
    parser.add_argument('--aneurysm-json', required=True,
                       help='Aneurysm location JSON file')
    parser.add_argument('--pattern', default='*_MRA*_seg*.nii.gz',
                       help='File pattern to match (default: *_MRA*_seg*.nii.gz)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum parallel workers')
    parser.add_argument('--smoothing-iterations', type=int, default=5,
                       help='Smoothing iterations')
    parser.add_argument('--tet-epsilon', type=float, default=1e-3,
                       help='fTetWild epsilon parameter')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip patients that already have output')
    parser.add_argument('--patient-list', help='Text file with specific patients to process')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(output_dir / "batch_processing.log")
    
    # Get files to process
    if args.patient_list:
        with open(args.patient_list, 'r') as f:
            patient_ids = [line.strip() for line in f if line.strip()]
        nifti_files = []
        for pid in patient_ids:
            files = list(data_dir.glob(f"{pid}*.nii.gz"))
            if files:
                nifti_files.append(files[0])
            else:
                logger.warning(f"Patient {pid} not found in {data_dir}")
    else:
        nifti_files = get_patient_files(data_dir, args.pattern)
    
    # Filter existing if requested
    if args.skip_existing:
        filtered_files = []
        for f in nifti_files:
            output_patient_dir = output_dir / f.stem
            if not (output_patient_dir / "pipeline_results.json").exists():
                filtered_files.append(f)
            else:
                logger.info(f"Skipping {f.stem} - already processed")
        nifti_files = filtered_files
    
    if not nifti_files:
        logger.warning("No files to process")
        return
    
    logger.info(f"Found {len(nifti_files)} files to process")
    
    # Prepare extra arguments
    extra_args = [
        "--smoothing-iterations", str(args.smoothing_iterations),
        "--tet-epsilon", str(args.tet_epsilon)
    ]
    
    # Run batch processing
    results_df = batch_process(
        nifti_files,
        output_dir,
        Path(args.aneurysm_json),
        max_workers=args.max_workers,
        extra_args=extra_args
    )
    
    # Save results summary
    results_df.to_csv(output_dir / "batch_processing_summary.csv", index=False)
    
    # Save failed cases for reprocessing
    failed_df = results_df[results_df['status'] != 'success']
    if len(failed_df) > 0:
        failed_df.to_csv(output_dir / "failed_cases.csv", index=False)
        logger.warning(f"Failed cases saved to: {output_dir}/failed_cases.csv")
    
    logger.info(f"\nBatch processing complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 