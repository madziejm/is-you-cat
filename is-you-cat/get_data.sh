#!/usr/bin/env bash
DIR="Dataset"
if [ -d "$DIR" ]; then
    echo "$DIR exists"
    exit 0
fi
mkdir $DIR
cd $DIR
echo "Downloading Dataset"
wget -q --show-progress https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
gdown https://drive.google.com/u/0/uc?id=10NIeg2v6b9SzBBkqzbxUT_xTTgzIjsmv
echo -ne "Unziping..."
mkdir NonCats
tar -xzf 101_ObjectCategories.tar.gz -C NonCats && rm 101_ObjectCategories.tar.gz
unzip -qq -j kagglecatsanddogs_*.zip 'PetImages/Cat/*' -d Cats && rm kagglecatsanddogs_*.zip
rm Cats/Thumbs.db Cats/666.jpg
echo " OK" 
exit 1


