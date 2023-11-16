FILE=$1
URL=http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/$FILE.zip
TAR_FILE=./$FILE.tar.gz
TARGET_DIR=./data/datasets/pix2pix/$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C ./
rm $TAR_FILE
