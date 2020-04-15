###############################################################################
# PART 1 : Visualisation de l'image
###############################################################################
from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# Importation unitaire de l'image
strImgPath = "C:/Users/Bernard/Documents/Python Scripts/Plate_Recognition_IMG/02. Collecte/02. ImageVeh/"
strImgName = "2019-01-28-17-18-29-071886.png"
car_image = imread(strImgPath + strImgName, as_grey=True)
# Normalement 2d array
print(car_image.shape)

# Visualisation rapide en noir et blanc de l'image
gray_car_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
ax2.imshow(binary_car_image, cmap="gray")
plt.show()

###############################################################################
# PART 2 : Redimension de l'image
###############################################################################
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization

# groups connected regions them together
label_image = measure.label(binary_car_image)
fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image, cmap="gray");

# properties
for region in regionprops(label_image):
    if region.area < 50:
        # si région trop petite, alors ce n'est probablement pas une plaque
        continue

    # the bounding box coordinates
    minRow, minCol, maxRow, maxCol = region.bbox
    rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)
    ax1.add_patch(rectBorder)
    # visualisation des rectangle de chacune des régions de l'image en rouge

plt.show()

###############################################################################
# PART 3 : Détection de la plaque sur l'image
###############################################################################
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization

# this gets all the connected regions and groups them together
label_image = measure.label(binary_car_image)

# getting the maximum width, height and minimum width and height that a license plate can be
plate_dimensions = (0.02*label_image.shape[0], 0.2*label_image.shape[0], 0.10*label_image.shape[1], 0.4*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []
plate_like_objects = []
fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image, cmap="gray");

# properties
for region in regionprops(label_image):
    if region.area < 50:
        # si région trop petite, alors ce n'est probablement pas une plaque
        continue

    # the bounding box coordinates
    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col
    # on vérifie les conditions de dimensions minimales adaptées pour une plaque d'immatriculation
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        plate_like_objects.append(binary_car_image[min_row:max_row,
                                  min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col,
                                              max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
    # On affiche les rectangles précédents avec le respect des dimensions d'une plaque

plt.show()

###############################################################################
# PART 4 : Redimension ++
###############################################################################
import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
#import cca2

# The invert was done so as to convert the black pixel to white pixel and vice versa
license_plate = np.invert(plate_like_objects[2])
#license_plate = np.invert(cca2.plate_like_objects[1])

labelled_plate = measure.label(license_plate)

fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")


# largeur entre 2% and 40%
# hauteur entre 60% and 90%
print(license_plate.shape[0], license_plate.shape[1])
character_dimensions = (0.60*license_plate.shape[0], 0.90*license_plate.shape[0], 0.02*license_plate.shape[1], 0.40*license_plate.shape[1])
min_height, max_height, min_width, max_width = character_dimensions

characters = []
counter=0
column_list = []
for regions in regionprops(labelled_plate):
    y0, x0, y1, x1 = regions.bbox
    region_height = y1 - y0
    region_width = x1 - x0

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
        roi = license_plate[y0:y1, x0:x1]

        # on dessine un rectangle pour isoler chacune des lettres de la plaque
        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                       linewidth=2, fill=False)
        ax1.add_patch(rect_border)

        # Redimension des lettres en 20X20
        resized_char = resize(roi, (28, 28))
        characters.append(resized_char)

        column_list.append(x0)

plt.show()




###############################################################################
# PART 5 : Redimension ++
###############################################################################

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

allItems = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
letters = [
            'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

numbers = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]


def read_training_data(training_directory, imgType):
    image_data = []
    target_data = []
    for each_letter in imgType:
        for each in range(1000):
            image_path = os.path.join(training_directory, each_letter, each_letter + '-' + str(each) + '.png')
            # read each image of each character
            img_details = imread(image_path, as_grey=True)
            # converts each character image to binary image
            binary_image = img_details < threshold_otsu(img_details)
            # the 2D array of each image is flattened because the machine learning
            # classifier requires that each sample is a 1D array
            # therefore the 20*20 image becomes 1*400
            # in machine learning terms that's 400 features with each pixel
            # representing a feature
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)

    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):

    accuracy_result = cross_val_score(model, train_data, train_label,
                                      cv=num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")

    print(accuracy_result * 100)


#current_dir = os.path.dirname(os.path.realpath(__file__))
current_dir = "C:/Users/Bernard/Documents/Python Scripts/Plate_Recognition_IMG/"
training_dataset_dir = os.path.join(current_dir, 'train')
training_dataset_dir2 = os.path.join(current_dir, '04. Train/HandWrittenData')

#############################
# Modèle global (nombres + lettres)
#############################
image_data, target_data = read_training_data(training_dataset_dir2, allItems)

# Utilisation d'un modele SVC en prédicition (optimiser plus tard avec CNN)
svc_model = SVC(kernel='linear', probability=True)

cross_validation(svc_model, 4, image_data, target_data)

# Train
svc_model.fit(image_data, target_data)

# Stockage du modèle pour le réutiliser ultérieurement
save_directory = os.path.join(current_dir, '03. Modeles/SVC/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, save_directory + '/svc_allItems.pkl')

#############################
# Modèle dédié aux lettres
#############################
image_data, target_data = read_training_data(training_dataset_dir2, letters)

# Utilisation d'un modele SVC en prédicition (optimiser plus tard avec CNN)
svc_model = SVC(kernel='linear', probability=True)

cross_validation(svc_model, 4, image_data, target_data)

# Train
svc_model.fit(image_data, target_data)

# Stockage du modèle pour le réutiliser ultérieurement
save_directory = os.path.join(current_dir, '03. Modeles/SVC/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, save_directory + '/svc_letters_2.pkl')

#############################
# Modèle dédié aux nombres
#############################
#image_data, target_data = read_training_data(training_dataset_dir, numbers)
image_data, target_data = read_training_data(training_dataset_dir2, numbers)

# Utilisation d'un modele SVC en prédicition (optimiser plus tard avec CNN)
svc_model = SVC(kernel='linear', probability=True)

cross_validation(svc_model, 4, image_data, target_data)

# Train
svc_model.fit(image_data, target_data)

# Stockage du modèle pour le réutiliser ultérieurement
save_directory = os.path.join(current_dir, '03. Modeles/SVC/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, save_directory + '/svc_numbers_2.pkl')



###############################################################################
# PART 6 : TEST SUR IMAGE
###############################################################################

import os
from sklearn.externals import joblib

current_dir = "C:/Users/Bernard/Documents/Python Scripts/Plate_Recognition_IMG/"

# Modélisation sur image avec modèle stocké pour reconnaitre la plaque
model_dir = os.path.join(current_dir, '03. Modeles/SVC/svc_allItems.pkl')
model = joblib.load(model_dir)


#classification_result = []
#for each_character in characters[0:2]:
#    # converts it to a 1D array
#    each_character = each_character.reshape(1, -1);
#    result = model_letters.predict(each_character)
#    classification_result.append(result)
#
#for each_character in characters[2:5]:
#    # converts it to a 1D array
#    each_character = each_character.reshape(1, -1);
#    result = model_numbers.predict(each_character)
#    classification_result.append(result)
#    
##classification_result = []
#for each_character in characters[5:7]:
#    # converts it to a 1D array
#    each_character = each_character.reshape(1, -1);
#    result = model_letters.predict(each_character)
#    classification_result.append(result)
    
classification_result = []
for each_character in characters:
    # converts it to a 1D array
    each_character = each_character.reshape(1, -1);
    result = model.predict(each_character)
    classification_result.append(result)
#    
##print(classification_result)

plate_string = ''
for eachPredict in classification_result:
    plate_string += eachPredict[0]

print(plate_string)

# Parfois les caractères de plate_string sont dans le mauvais ordre donc on les trie

column_list_copy = column_list[:]
column_list.sort()
rightplate_string = ''
for each in column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print(rightplate_string)
