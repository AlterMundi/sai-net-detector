#!/bin/bash
#
# Master script to download and prepare all datasets for SAI-Net YOLOv8 Detector training.
# Downloads PyroSDIS and FASDD datasets, converts them to YOLO format.
#
# Usage:
#   bash scripts/download_datasets.sh [options]
#
# Options:
#   --pyrosdis-only    Download only PyroSDIS dataset
#   --fasdd-only       Download only FASDD dataset  
#   --fasdd-id         Specify FASDD Kaggle dataset ID
#   --skip-convert     Skip FASDD to YOLO conversion
#   --help             Show this help message
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
DOWNLOAD_PYROSDIS=true
DOWNLOAD_FASDD=true
FASDD_DATASET_ID=""
SKIP_CONVERT=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pyrosdis-only)
            DOWNLOAD_PYROSDIS=true
            DOWNLOAD_FASDD=false
            shift
            ;;
        --fasdd-only)
            DOWNLOAD_PYROSDIS=false
            DOWNLOAD_FASDD=true
            shift
            ;;
        --fasdd-id)
            FASDD_DATASET_ID="$2"
            shift 2
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --help)
            head -n 15 "$0" | grep '^#' | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Helper functions
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is required but not installed"
        return 1
    fi
    print_success "Python3 available: $(python3 --version)"
}

check_dependencies() {
    print_header "Checking Dependencies"
    
    check_python
    
    # Check required Python packages
    local required_packages=("datasets" "pillow" "numpy")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_warning "Missing Python packages: ${missing_packages[*]}"
        echo "Installing missing packages..."
        pip3 install "${missing_packages[@]}"
    else
        print_success "All required packages available"
    fi
    
    # Check Kaggle API if downloading FASDD
    if [[ "$DOWNLOAD_FASDD" == true ]]; then
        if ! python3 -c "import kaggle" 2>/dev/null; then
            print_warning "Kaggle package not found"
            echo "Installing kaggle package..."
            pip3 install kaggle
        fi
        
        # Check Kaggle credentials
        if ! kaggle --version &>/dev/null; then
            print_error "Kaggle API not properly configured"
            echo "Please setup Kaggle credentials:"
            echo "1. Get API key from kaggle.com/account"
            echo "2. Place kaggle.json in ~/.kaggle/"
            echo "3. Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
            return 1
        fi
        print_success "Kaggle API configured"
    fi
}

create_directories() {
    print_header "Creating Directory Structure"
    
    mkdir -p data/raw/pyro-sdis/{images,labels}/{train,val}
    mkdir -p data/raw/fasdd
    mkdir -p data/yolo/{images,labels}/{train,val,test}
    mkdir -p runs/detect
    
    print_success "Directory structure created"
}

download_pyrosdis() {
    print_header "Downloading PyroSDIS Dataset"
    
    if [[ -d "data/raw/pyro-sdis/images" ]] && [[ $(find data/raw/pyro-sdis/images -name "*.jpg" | wc -l) -gt 100 ]]; then
        print_warning "PyroSDIS dataset already exists"
        read -p "Redownload? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_success "Skipping PyroSDIS download"
            return 0
        fi
    fi
    
    echo "Downloading from Hugging Face: pyronear/pyro-sdis"
    python3 scripts/export_pyrosdis_to_yolo.py \
        --hf_repo pyronear/pyro-sdis \
        --output data/raw/pyro-sdis \
        --splits train val \
        --single-cls
    
    if [[ $? -eq 0 ]]; then
        print_success "PyroSDIS dataset downloaded and converted"
    else
        print_error "Failed to download PyroSDIS dataset"
        return 1
    fi
}

download_fasdd() {
    print_header "Downloading FASDD Dataset"
    
    if [[ -d "data/raw/fasdd/images" ]] && [[ $(find data/raw/fasdd/images -name "*.jpg" | wc -l) -gt 1000 ]]; then
        print_warning "FASDD dataset already exists"
        read -p "Redownload? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_success "Skipping FASDD download"
            return 0
        fi
    fi
    
    local fasdd_cmd="python3 scripts/download_fasdd.py --output data/raw/fasdd"
    
    if [[ -n "$FASDD_DATASET_ID" ]]; then
        fasdd_cmd="$fasdd_cmd --dataset-id $FASDD_DATASET_ID"
        echo "Using specified FASDD dataset ID: $FASDD_DATASET_ID"
    else
        echo "Auto-detecting FASDD dataset on Kaggle..."
    fi
    
    eval $fasdd_cmd
    
    if [[ $? -eq 0 ]]; then
        print_success "FASDD dataset downloaded"
    else
        print_error "Failed to download FASDD dataset"
        echo "You may need to:"
        echo "1. Accept the dataset terms on Kaggle website"
        echo "2. Provide the correct dataset ID with --fasdd-id"
        return 1
    fi
}

convert_fasdd_to_yolo() {
    if [[ "$SKIP_CONVERT" == true ]]; then
        print_warning "Skipping FASDD conversion (--skip-convert specified)"
        return 0
    fi
    
    print_header "Converting FASDD to YOLO Format"
    
    if [[ -d "data/yolo/images" ]] && [[ $(find data/yolo/images -name "*.jpg" | wc -l) -gt 1000 ]]; then
        print_warning "YOLO dataset already exists"
        read -p "Reconvert? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_success "Skipping FASDD conversion"
            return 0
        fi
    fi
    
    echo "Converting FASDD from COCO to YOLO format..."
    python3 scripts/convert_fasdd_to_yolo.py \
        --src data/raw/fasdd \
        --dst data/yolo \
        --map-classes smoke
    
    if [[ $? -eq 0 ]]; then
        print_success "FASDD converted to YOLO format"
    else
        print_error "Failed to convert FASDD to YOLO format"
        return 1
    fi
}

validate_datasets() {
    print_header "Validating Datasets"
    
    local all_valid=true
    
    # Validate PyroSDIS
    if [[ "$DOWNLOAD_PYROSDIS" == true ]]; then
        if [[ -d "data/raw/pyro-sdis/images/train" ]] && [[ -f "data/raw/pyro-sdis/data.yaml" ]]; then
            local pyro_count=$(find data/raw/pyro-sdis/images -name "*.jpg" | wc -l)
            print_success "PyroSDIS: $pyro_count images"
        else
            print_error "PyroSDIS validation failed"
            all_valid=false
        fi
    fi
    
    # Validate FASDD
    if [[ "$DOWNLOAD_FASDD" == true ]]; then
        if [[ -d "data/raw/fasdd/images" ]] && [[ -d "data/raw/fasdd/annotations" ]]; then
            local fasdd_count=$(find data/raw/fasdd/images -name "*.jpg" | wc -l)
            print_success "FASDD (raw): $fasdd_count images"
        else
            print_error "FASDD validation failed"
            all_valid=false
        fi
        
        # Validate YOLO conversion
        if [[ "$SKIP_CONVERT" == false ]]; then
            if [[ -d "data/yolo/images" ]] && [[ -f "data/yolo/data.yaml" ]]; then
                local yolo_count=$(find data/yolo/images -name "*.jpg" | wc -l)
                print_success "FASDD (YOLO): $yolo_count images"
            else
                print_error "FASDD YOLO conversion validation failed"
                all_valid=false
            fi
        fi
    fi
    
    if [[ "$all_valid" == true ]]; then
        print_success "All datasets validated successfully"
    else
        print_error "Some datasets failed validation"
        return 1
    fi
}

show_next_steps() {
    print_header "Next Steps"
    
    echo "Datasets ready for training! Available configurations:"
    echo
    echo "🔥 Stage 1 - FASDD Pre-training (Multi-class):"
    echo "   python scripts/train_two_stage.py --stage 1 --test-mode --epochs 3"
    echo "   python scripts/train_two_stage.py --stage 1  # Full training"
    echo
    echo "💨 Stage 2 - PyroSDIS Fine-tuning (Single-class):"
    echo "   python scripts/train_two_stage.py --stage 2 --test-mode --epochs 3" 
    echo "   python scripts/train_two_stage.py --stage 2  # Full training"
    echo
    echo "⚡ Complete Two-Stage Workflow:"
    echo "   python scripts/train_two_stage.py --full-workflow"
    echo
    echo "📊 Direct Training:"
    echo "   python scripts/train_detector.py --config optimal"
    echo
    echo "For more information, see:"
    echo "- CLAUDE.md for detailed training instructions"
    echo "- docs/planentrenamientoyolov8.md for two-stage methodology"
}

main() {
    print_header "SAI-Net Detector Dataset Download"
    echo "YOLOv8 Detector Development Repository"
    
    # Check what will be downloaded
    if [[ "$DOWNLOAD_PYROSDIS" == true ]] && [[ "$DOWNLOAD_FASDD" == true ]]; then
        echo "Mode: Download both PyroSDIS and FASDD datasets"
    elif [[ "$DOWNLOAD_PYROSDIS" == true ]]; then
        echo "Mode: Download PyroSDIS only"
    elif [[ "$DOWNLOAD_FASDD" == true ]]; then
        echo "Mode: Download FASDD only"
    fi
    
    # Execute pipeline
    check_dependencies || exit 1
    create_directories || exit 1
    
    if [[ "$DOWNLOAD_PYROSDIS" == true ]]; then
        download_pyrosdis || exit 1
    fi
    
    if [[ "$DOWNLOAD_FASDD" == true ]]; then
        download_fasdd || exit 1
        convert_fasdd_to_yolo || exit 1
    fi
    
    validate_datasets || exit 1
    show_next_steps
    
    print_success "Dataset download and preparation complete!"
}

# Run main function
main "$@"