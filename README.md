# Galaxy Morphology "Classification" (Project 4)

Using a Convolution Neural Network (CNN), we trained a model on tens of thousands of images to make predictions on what people would classify the galaxy as.

## Group Members
- Zachary Aaronson
    - Data and model
- Kali Schoenrock
    - Presentation
- Jason Stone
    - Visualizations

## Setup
### Data Sources
Data and images were collected from the Galaxy Zoo 2 project which uses images from the Sloan Digital Sky Survey.

Links to files that must be downloaded:
- [https://data.galaxyzoo.org/#section-12](https://data.galaxyzoo.org/#section-12)
    - gz2_hart16.csv (364 MB)
- [https://zenodo.org/record/3565489#.Y3vFKS-l0eY](https://zenodo.org/record/3565489#.Y3vFKS-l0eY)
    - gz2_filename_mapping.csv (13 MB)
    - images_gz2.zip (3.4 GB)

### File Structure
Many files are used and generated that are not included in this repository due to size. The below file structure shows where downloaded files should be placed and where generated files will be created.

```
Galaxy_Morphology_Classification
├── 📁 data
│   ├── 📁 images
│   │   └── *.jpg (extracted files from images_gz2.zip)
│   ├── 📁 images_processed
│   │   └── *.png (empty after splitting test and train)
│   ├── 📁 model
│   │   ├── 📁 checkpoints
│   │   ├── 🌠 GalaxyConfidenceModel.keras
│   │   └── 📗 training_log.csv
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

### Packages
- Matplotlib
- Numpy
- OpenCV
- Pandas
- Scikit-learn
- Scipy
- Tensorflow/Keras

## Preprocessing Data and Images
Data and images processing code can be found in [data_image_cleaning.ipynb](data_image_cleaning.ipynb).

### Data
The data from the two csv files (`gz2_filename_mapping.csv` and `gz2_hart16.csv`) are loaded with the correct data types. Most of the columns from hart16 are skipped as we only want a few interesting columns and the 37 '_debiased' columns that will be the `y` for training our model. Duplicates are found and removed, rows with null values are removed, and the data is merged.

Next, the `asset_id` column values are compared to the file names of all the JPG the images in `data\images\`. Rows that do not have a corresponding image are removed and images that do not have a row of data will not be processed in later steps.

Following that, we then create a new column, `class_reduced`, that will be used to stratify our data for testing and training to make sure we have a proportional distribution of rarer classes in both data sets. The value from the `gz2_class` is used to create the values for this new column, with very rare classes being combined into more general classes and super rare classes being placed into a single group.

Finally, we save our 239,267 row, 45 column DataFrame to a SQLite database.

### Images
The original images found in `images_gz2.zip` and extracted to `data\images\` are 424×424 color jpg files. A function was written to find the central feature and crop and/or scale the down the image into a 106×106 grayscale png files using the following steps.

1. Read image with OpenCV from JPG
2. Convert to grayscale
3. Gaussian Blur
    - smooths the image a small amount to remove some noise
4. Threshold
    - As images of space have a near black background, this is a simple step that makes features obvious for later steps
5. Dilate threshold
    - slightly grow the all white regions to remove small holes and pad the features
6. Find Contour with a minimum area with center closest to center of image
    - Finds the distinct features and isolates the one in the center of the image which is the galaxy the data refers to.
7. Get bounding box of contour
    - Finds the full size of the detected feature
8. Check if bounding box fully contained inside target rectangles
    - This step determines what the crop/scale step does
9. Crop/Scale
    - If inside smallest rectangle crop only
    - If inside second rectangle crop, then scale
    - Otherwise, scale
10. Save the final grayscale image as PNG

Two examples of this process are show below:

![Example of image processing 1](/images/process_image_ex_1.png)
![Example of image processing 2](/images/process_image_ex_2.png)

## Model
We used a Convolution Neural Network created with Keras and Tensorflow. A slightly modified section of the code is shown below with the different layers, compiling the model and fitting the model. The full code can be found in [model_training.ipynb](model_training.ipynb).

The model takes in 106×106 grayscale (single channel) png files that have been converted to Numpy float32 arrays with values in the range [0, 1].

The choices for this model were made to balance _training speed_ vs _correctness_ as it was run on a single laptop on the CPU.

```py
model = Sequential()

# Add convolution layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(106, 106, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2, seed=RANDOM_STATE))

# Flatten the output from convolution layers
model.add(Flatten())

# Add dense (fully connected) layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2, seed=RANDOM_STATE))

model.add(Dense(64, activation='relu'))

# Add the output layer with 37 units (for 37 classes)
model.add(Dense(37, activation='sigmoid'))

# --- --- --- ---
# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError(), r2_score])

# Train the model
model.fit(X_train_images,
          y_train,
          epochs=EPOCHS,
          callbacks=callbacks_,
          batch_size=2_000,
          validation_split=0.1)
```

### Callbacks
A variety of callbacks were used during training for various purposes.
- ModelCheckpoint
    - Saves the weights every epoch
- EarlyStopping
    - End the training if _loss_ is stagnant for 6 epochs
- LearningRateScheduler
    - Reduce the learning rate every 10 epochs by 0.8
- ReduceLROnPlateau
    - Reduce the learning rate by a factor of 10 if _loss_ plateaus for 4 epochs
- CSVLogger
    - Save information on each epoch to a CSV file, continue the file if training is continued later

## Results
The model was trained for XYZ epochs which took XYZ hr:m:s.

After evaluating the model with 59,817 test images the final values were:
- Loss (Mean Squared Error): 0._-__
- Root Mean Squared Error: 0._-__

![Metrics over training time]()

## Example Images
Hand selected images that best represent each class.

![Sample images for each of the 37 classes](images/example_samples.png)

## Citations
- Willett et al. (2013, MNRAS, 435, 2835, DOI: [10.1093/mnras/stt1458](https://doi.org/10.1093/mnras/stt1458))
    - Galaxy Zoo 2
- Hart et al. (2016, MNRAS, 461, 3663, DOI: [10.1093/mnras/stw1588](https://doi.org/10.1093/mnras/stw1588))
    - Debiased data
- Sky Map [https://in-the-sky.org/data/constellations_map.php](https://in-the-sky.org/data/constellations_map.php?latitude=37.1305&longitude=-113.5083&timezone=-07%3A00)