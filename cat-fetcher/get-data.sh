#!/bin/bash


DIR="Cats_Dataset"
if [ -d "$DIR" ]; then
    echo "$DIR exists"
    exit 0
fi
mkdir $DIR
cd $DIR
echo "Downloading Dataset"
wget -q --show-progress https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
echo -ne "Unziping..."
unzip -qq -j kagglecatsanddogs_*.zip 'PetImages/Cat/*' -d Images && rm kagglecatsanddogs_*.zip
rm Images/Thumbs.db 
echo " OK" 
echo -ne "Generating txt file"
touch images.txt
ls Images > images.txt
echo " OK"
exit 1
