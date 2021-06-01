#!/usr/bin/env bash

pip install openimages
oi_download_images --base_dir temp --limit 26 --labels "Rose" "Flag" "Flashlight" "Sea turtle" "Camera" "Animal" "Glove" "Crocodile" "Cattle" "House" "Guacamole" "Penguin" "Vehicle registration plate" "Bench" "Ladybug" "Human nose" "Watermelon" "Flute" "Butterfly" "Washing machine" "Raccoon" "Segway" "Taco" "Jellyfish" "Cake" "Pen" "Cannon" "Bread" "Tree" "Shellfish" "Bed" "Hamster" "Hat" "Toaster" "Sombrero" "Tiara" "Bowl" "Dragonfly" "Moths and butterflies" "Antelope" "Vegetable" "Torch"
mkdir niekotki
find . -wholename "./temp/*/*.jpg" -exec cp {} niekotki/ \;
rm -rf temp
wget -q --show-progress https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
unzip -qq -j kagglecatsanddogs_*.zip 'PetImages/Cat/*' -d temp && rm kagglecatsanddogs_*.zip
rm temp/666.jpg
rm temp/Thumbs.db
python3 split-cats.py
mkdir Train Val Test
mv cats_1 Train/Cat
mv cats_2 Val/Cat
mv cats_3 Test/Cat
mv niekotki Test/NCat
