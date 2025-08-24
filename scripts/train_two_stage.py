#!/usr/bin/env python3
"""
SAI-Net Two-Stage Training Workflow Script

Implements the complete two-stage training approach:
1. Stage 1: FASDD multi-class pre-training (fire + smoke)
2. Stage 2: PyroSDIS single-class fine-tuning (smoke only)

Based on the methodology from planentrenamientoyolov8.md with
validated hardware configuration (2×A100, batch=60, workers=8, 1440×1440).
"""

import sys
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector.train import (
    train_stage1_fasdd, 
    train_stage2_pyrosdis, 
    train_two_stage_workflow
)

def main():
    parser = argparse.ArgumentParser(
        description="SAI-Net Two-Stage Training Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete two-stage workflow
  python scripts/train_two_stage.py --full-workflow
  
  # Run only Stage 1 (FASDD pre-training)
  python scripts/train_two_stage.py --stage 1
  
  # Run only Stage 2 with custom checkpoint
  python scripts/train_two_stage.py --stage 2 --checkpoint path/to/stage1/best.pt
  
  # Test Stage 1 with short run
  python scripts/train_two_stage.py --stage 1 --test-mode --epochs 3
        """
    )
    
    parser.add_argument("--full-workflow", action="store_true",
                       help="Execute complete two-stage workflow (Stage 1 + Stage 2)")
    parser.add_argument("--stage", type=int, choices=[1, 2],
                       help="Run specific stage only (1: FASDD, 2: PyroSDIS)")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to Stage 1 checkpoint for Stage 2 (auto-detected if not provided)")
    parser.add_argument("--test-mode", action="store_true",
                       help="Test mode with reduced epochs for validation")
    parser.add_argument("--epochs", type=int,
                       help="Override default epochs (for testing)")
    parser.add_argument("--non-interactive", action="store_true",
                       help="Disable interactive prompts")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.full_workflow, args.stage]):
        print("❌ Error: Must specify either --full-workflow or --stage")
        parser.print_help()
        sys.exit(1)
    
    if args.stage == 2 and not args.checkpoint and not Path("/dev/shm/rrn/sai-net-detector/runs/sai_stage1_fasdd_multiclass/weights/best.pt").exists():
        print("❌ Error: Stage 2 requires Stage 1 checkpoint. Run Stage 1 first or provide --checkpoint")
        sys.exit(1)
    
    print("🚀 SAI-Net Two-Stage Training Workflow")
    print("=" * 60)
    print("📊 Configuration Summary:")
    print(f"   Hardware: 2×A100-40GB DDP, 1440×1440 resolution")
    print(f"   Batch size: 60 (30×2 GPUs), Workers: 8")
    print(f"   Cache: RAM (~350GB), Mixed precision: AMP")
    print(f"   Datasets: FASDD (~95K) → PyroSDIS (~33K)")
    
    if args.test_mode:
        print(f"   ⚠️  TEST MODE: Reduced epochs for validation")
    
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        if args.full_workflow:
            # Complete two-stage workflow
            print("\n🔄 Executing complete two-stage workflow...")
            
            # Override function for test mode
            if args.test_mode or args.epochs:
                print("⚠️  Test mode not supported for full workflow")
                print("💡 Use --stage 1 --test-mode for Stage 1 testing")
                sys.exit(1)
            
            results = train_two_stage_workflow()
            
            # Final report
            print("\n" + "="*60)
            print("📋 FINAL WORKFLOW REPORT")
            print("="*60)
            
            if results['workflow_success']:
                print("✅ Two-stage workflow completed successfully!")
                print(f"📁 Final model: {results['final_model_path']}")
                print("\n📊 Stage Results:")
                print(f"   Stage 1 (FASDD): {results['stage1_results']['save_dir']}")
                print(f"   Stage 2 (PyroSDIS): {results['stage2_results']['save_dir']}")
            else:
                print(f"❌ Workflow failed: {results.get('error', 'Unknown error')}")
                if results.get('stage1_results') and results['stage1_results']['success']:
                    print(f"💡 Stage 1 succeeded: {results['stage1_results']['save_dir']}")
                    print("   You can retry Stage 2 separately")
        
        elif args.stage == 1:
            # Stage 1 only
            print("\n📊 STAGE 1: FASDD Multi-class Pre-training")
            print("🎯 Objective: Learn diverse fire/smoke detection")
            if not args.test_mode:
                print("📈 Expected time: ~39 hours (140 epochs)")
            print("-" * 40)
            
            # Test mode modifications
            if args.test_mode or args.epochs:
                print("⚠️  Running in test mode - creating modified function")
                from detector.train import train_detector
                
                test_epochs = args.epochs if args.epochs else 3
                print(f"🧪 Test epochs: {test_epochs}")
                
                results = train_detector(
                    data_yaml="configs/yolo/fasdd_stage1.yaml",
                    model="yolov8s.pt",
                    imgsz=1440,
                    epochs=test_epochs,  # Override epochs
                    batch=60,
                    device="0,1",
                    workers=8,
                    name=f"sai_stage1_fasdd_test_{test_epochs}ep",
                    single_cls=False,
                    interactive=not args.non_interactive
                )
            else:
                results = train_stage1_fasdd()
            
            if results['success']:
                print("✅ Stage 1 completed successfully!")
                print(f"📁 Checkpoint: {results['save_dir']}/weights/best.pt")
                print("💡 Ready for Stage 2 fine-tuning")
            else:
                print(f"❌ Stage 1 failed: {results['error']}")
                sys.exit(1)
        
        elif args.stage == 2:
            # Stage 2 only
            print("\n📊 STAGE 2: PyroSDIS Single-class Fine-tuning")
            print("🎯 Objective: Specialize for smoke-only detection")
            if not args.test_mode:
                print("📈 Expected time: ~8 hours (60 epochs)")
            print("-" * 40)
            
            checkpoint_path = args.checkpoint
            if checkpoint_path:
                print(f"📁 Using checkpoint: {checkpoint_path}")
            else:
                checkpoint_path = "/dev/shm/rrn/sai-net-detector/runs/sai_stage1_fasdd_multiclass/weights/best.pt"
                print(f"📁 Auto-detected checkpoint: {checkpoint_path}")
            
            # Test mode modifications
            if args.test_mode or args.epochs:
                print("⚠️  Running in test mode - creating modified function")
                from detector.train import train_detector
                
                test_epochs = args.epochs if args.epochs else 3
                print(f"🧪 Test epochs: {test_epochs}")
                
                results = train_detector(
                    data_yaml="configs/yolo/pyro_stage2.yaml",
                    model=checkpoint_path,
                    imgsz=1440,
                    epochs=test_epochs,  # Override epochs
                    batch=60,
                    device="0,1",
                    workers=8,
                    lr0=0.001,  # Fine-tuning LR
                    name=f"sai_stage2_pyrosdis_test_{test_epochs}ep",
                    single_cls=True,
                    interactive=not args.non_interactive
                )
            else:
                results = train_stage2_pyrosdis(checkpoint_path)
            
            if results['success']:
                print("✅ Stage 2 completed successfully!")
                print(f"📁 Final detector: {results['save_dir']}/weights/best.pt")
                print("🎯 Ready for deployment!")
            else:
                print(f"❌ Stage 2 failed: {results['error']}")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
    
    finally:
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        print(f"\n⏱️  Total execution time: {hours}h {minutes}m")

if __name__ == "__main__":
    main()