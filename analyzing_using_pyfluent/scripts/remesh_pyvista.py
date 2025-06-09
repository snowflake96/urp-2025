#!/usr/bin/env python3
"""
remesh_pyvista.py: Reduce triangle count of an STL file while preserving shape using PyVista.

Usage:
    python remesh_pyvista.py input.stl output.stl [--reduction R] [--smooth_iter N] [--relax F]

Options:
    --reduction R    Fraction of triangles to remove (0 < R < 1). Default: 0.5 (50% reduction)
    --smooth_iter N  Number of smoothing iterations after decimation. Default: 20
    --relax F        Relaxation factor for smoothing (0 < F < 1). Default: 0.1
"""

import argparse
import sys
import os
import pyvista as pv

def decimate_stl(input_stl: str, output_stl: str, reduction: float,
                 smooth_iter: int, relax: float):
    # Load the mesh
    mesh = pv.read(input_stl)
    print(f"Original triangles: {mesh.n_cells:,}")

    # Decimate while preserving topology
    decimated = mesh.decimate_pro(reduction=reduction, preserve_topology=True)
    print(f"Post-decimation triangles: {decimated.n_cells:,}")

    # Smooth to improve surface quality
    decimated = decimated.smooth(n_iter=smooth_iter, relaxation_factor=relax)
    print(f"Post-smoothing triangles: {decimated.n_cells:,}")

    # Save the reduced mesh
    decimated.save(output_stl)
    print(f"✅ Decimated mesh saved to: {output_stl}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reduce triangle count of an STL while preserving shape."
    )
    parser.add_argument("input_stl", help="Path to the input STL file")
    parser.add_argument("output_stl", help="Path to the output STL file")
    parser.add_argument(
        "--reduction", type=float, default=0.5,
        help="Fraction of triangles to remove (default: 0.5)"
    )
    parser.add_argument(
        "--smooth_iter", type=int, default=20,
        help="Number of smoothing iterations (default: 20)"
    )
    parser.add_argument(
        "--relax", type=float, default=0.1,
        help="Relaxation factor for smoothing (default: 0.1)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.isfile(args.input_stl):
        print(f"Error: Input STL not found: {args.input_stl}", file=sys.stderr)
        sys.exit(1)
    # Perform decimation and smoothing
    decimate_stl(
        args.input_stl,
        args.output_stl,
        args.reduction,
        args.smooth_iter,
        args.relax
    )

if __name__ == "__main__":
    main()