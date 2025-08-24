#!/usr/bin/env python3
"""
Test script for ForcedDDP training
Quick 1-epoch test to validate DDP functionality
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector.train import train_detector_forced_ddp

def main():
    parser = argparse.ArgumentParser(description="Test ForcedDDP Training")
    
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for test")
    parser.add_argument("--batch", type=int, default=60, help="Batch size (conservative for test)")
    parser.add_argument("--device", default="0,1", help="GPU devices")
    parser.add_argument("--name", default="forceddp_test_1epoch", help="Experiment name")
    parser.add_argument("--cache", default="ram", help="Cache mode")
    parser.add_argument("--non-interactive", action="store_true", help="Disable interactive prompts")
    
    args = parser.parse_args()
    
    print("🧪 Starting ForcedDDP Test Training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch: {args.batch}")
    print(f"   Device: {args.device}")
    print(f"   Cache: {args.cache}")
    print("=" * 50)
    
    # Run test training
    results = train_detector_forced_ddp(
        data_yaml="configs/yolo/pyro_fasdd.yaml",
        model="yolov8s.pt",
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=8,  # Conservative for testing
        name=args.name,
        cache=args.cache,
        imgsz=[1440, 808],
        interactive=not args.non_interactive  # Allow disabling prompts
    )
    
    # Report results
    print("=" * 50)
    print("📊 TEST RESULTS:")
    
    if results['success']:
        print("✅ ForcedDDP test SUCCESSFUL!")
        print(f"   DDP Mode: {'Enabled' if results['ddp_mode'] else 'Single GPU Fallback'}")
        print(f"   Save Dir: {results['save_dir']}")
        if results.get('best_fitness'):
            print(f"   Best Fitness: {results['best_fitness']:.4f}")
        
        # Recommendation
        if results['ddp_mode']:
            print("\n💡 RECOMMENDATION: DDP working! Safe to use for full training")
        else:
            print("\n💡 RECOMMENDATION: DDP fallback to single GPU occurred")
            
    else:
        print("❌ ForcedDDP test FAILED!")
        print(f"   Error: {results['error']}")
        print(f"   DDP Attempted: {results.get('ddp_mode', 'Unknown')}")
        
        if results.get('recommendation'):
            print(f"\n💡 RECOMMENDATION: {results['recommendation']}")
        
        print("\n🔄 NEXT STEP: Switch to single GPU solution")
        sys.exit(1)

if __name__ == "__main__":
    main()