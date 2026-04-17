# SafeGear AI Assets

This directory contains demo assets for the SafeGear AI project.

## Directory Structure

```
assets/
├── README.md           # This file
├── demo_banner.png     # Main project banner/image
├── sample_detections/  # Sample detection outputs
│   ├── safe_worker.jpg
│   ├── violation_helmet.jpg
│   └── violation_vest.jpg
└── icons/              # UI icons and graphics
    ├── helmet_icon.png
    ├── vest_icon.png
    └── logo.png
```

## Adding Custom Assets

### Demo Banner
Replace `demo_banner.png` with your own project banner (recommended: 1280x400 pixels)

### Sample Images
Add sample detection images to showcase your model's capabilities:
- Safe scenarios (all PPE worn correctly)
- Violation scenarios (missing helmet, vest, etc.)
- Different environments (construction, road, industrial)

### Icons
Custom icons for:
- Helmet detection
- Safety vest detection
- Mask detection
- Violation alerts
- Application logo

## Image Requirements

- Format: PNG or JPG
- Safe detection images: Use green highlights
- Violation images: Use red highlights
- Resolution: Minimum 640x480 for clarity
