import os
from PIL import Image

# Chemin des dossiers
cat_folder = "..\\dataset\\cats"
dog_folder = "..\\dataset\\dogs"

# Affichage de quelques images
for i, image_name in enumerate(os.listdir(dog_folder)[:18]):
    img = Image.open(os.path.join(dog_folder, image_name))
    img.show()
