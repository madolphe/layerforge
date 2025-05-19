#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from layerforge.dataset_generator import DatasetGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate a dataset of multilayer optical structures and their spectra.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the configuration file (default: config.yaml)'
    )
    return parser.parse_args()

def main():
    """Main entry point for the dataset generation script."""
    args = parse_args()
    config_path = Path(args.config)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    logger.info(f"Using configuration from: {config_path}")
    generator = DatasetGenerator(str(config_path))
    generator.generate_dataset()

if __name__ == "__main__":
    main() 