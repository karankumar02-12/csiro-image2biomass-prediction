# ----- Data/README.md -----
# Data Directory

## Download Instructions

### Using Kaggle API

1. Install the Kaggle API:
```bash
pip install kaggle
```

2. Set up your Kaggle credentials:
   - Go to https://www.kaggle.com/settings
   - Create New API Token
   - Place `kaggle.json` in `~/.kaggle/`

3. Download the competition data:
```bash
kaggle competitions download -c csiro-biomass
unzip csiro-biomass.zip -d data/
```

### Manual Download

1. Visit: https://www.kaggle.com/competitions/csiro-biomass/data
2. Download all files
3. Extract to this directory

## Expected Structure

After extraction, your data directory should look like:

```
data/
├── train.csv
├── test.csv
├── sample_submission.csv
├── train_images/
│   ├── ID1001187975.jpg
│   ├── ID1001187976.jpg
│   └── ...
└── test_images/
    ├── ID1001187975.jpg
    └── ...
```

## Data Description

### train.csv
- `image_path`: Path to image file
- `target_name`: One of [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
- `target`: Biomass value in grams

### test.csv
- `sample_id`: Unique identifier (image_id + target_name)
- `image_path`: Path to image file
- `target_name`: Target to predict

### Images
- Resolution: Variable (resized to 128x128 in preprocessing)
- Format: JPEG
- Content: Top-down pasture photographs
