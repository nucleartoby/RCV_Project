#!/usr/bin/env python3
# ""
import os
import argparse
import logging
from datetime import datetime

from Machine_Learning_Models.src.main.main import main as run_pipeline

def setup_logging(log_level, log_file=None):

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            filename=log_file
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run NASDAQ prediction pipeline")
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='Machine_Learning_Models/output',
        help='Directory to save model outputs and predictions'
    )
    
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--log-file', 
        type=str, 
        help='Log file path (if not specified, log to console)'
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Get current timestamp for run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    print(f"Starting NASDAQ prediction pipeline run {timestamp}")
    print(f"Output will be saved to: {run_output_dir}")
    
    # Run the pipeline
    results = run_pipeline()
    
    if results:
        print("\nPipeline completed successfully!")
        print(f"Model saved to: {results['model_file']}")
        print(f"Next day prediction: {results['prediction']:.2f} (Change: {results['predicted_change']:.2f}%)")
    else:
        print("\nPipeline execution failed. Check logs for details.")

if __name__ == "__main__":
    main()