# Galaxy Morphology Classification (Project 4)

Using a Convolution Neural Network (CNN), we trained a model on tens of thousands of images to detect the shape and features (morphology) of a galaxy from an image.

## Group Members
- Zachary Aaronson
    - Data and model
- Kali Schoenrock
    - Presentation
- Jason Stone
    - Visualizations

## Data Sources
Data and images were collected from the Galaxy Zoo 2 project which uses images from the Sloan Digital Sky Survey.

Links to files that must be downloaded:
- [https://data.galaxyzoo.org/#section-12](https://data.galaxyzoo.org/#section-12)
    - gz2_hart16.csv (364 MB)
- [https://zenodo.org/record/3565489#.Y3vFKS-l0eY](https://zenodo.org/record/3565489#.Y3vFKS-l0eY)
    - gz2_filename_mapping.csv (13 MB)
    - images_gz2.zip (3.4 GB)

## File Structure
```
Galaxy_Morphology_Classification
├── 📁 data
│   ├── 📁 images
│   │   └── *.jpg (extracted files from images_gz2.zip)
│   ├── 📁 images_processed
│   │   └── *.png (empty after splitting test and train)
│   ├── 📁 model
│   │   ├── 📁 checkpoints
│   │   └── 🌠 GalaxyConfidenceModel.keras
│   ├── 📁 test_images
│   │   └── *.png
│   ├── 📁 train_images
│   │   └── *.png
│   ├── 📖 galaxy_data.sqlite
│   ├── 📗 gz2_filename_mapping.csv
│   └── 📗 gz2_hart16.csv
├── 📁 images
├── 📔 *.ipynb (3 files)
├── 📄 README.md 
└── ⚙️ .gitignore
```

## Packages
- Matplotlib
- Numpy
- OpenCV
- Pandas
- Scikit-learn
- Tensorflow/Keras

## Citations
- Willett et al. (2013, MNRAS, 435, 2835, DOI: [10.1093/mnras/stt1458](https://doi.org/10.1093/mnras/stt1458))
    - Galaxy Zoo 2
- Hart et al. (2016, MNRAS, 461, 3663, DOI: [10.1093/mnras/stw1588](https://doi.org/10.1093/mnras/stw1588))
    - Debiased data
- Sky Map [https://in-the-sky.org/data/constellations_map.php](https://in-the-sky.org/data/constellations_map.php?latitude=37.1305&longitude=-113.5083&timezone=-07%3A00)