# Dataset Info

**Path**: `C:\Users\ANJANA\Downloads\data\data`

## Classes (35 total)
- Digits: `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
- Letters: `A` through `Z`

## Structure
Each class has its own sub-folder containing ~1,200 JPEG images.

```
data/
├── A/   (1200 images)
├── B/   (1200 images)
...
├── Z/   (1200 images)
├── 1/   (1200 images)
...
└── 9/   (1200 images)
```

## Notes
- Images vary in background and lighting (good for generalization)
- Dataset stays in the Downloads folder; config.py points to it
