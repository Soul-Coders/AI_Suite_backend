sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev

mkdir model
cd model

wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt
curl -LJO https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt


wget --no-check-certificate --content-disposition https://github.com/richzhang/colorization/blob/a1642d6ac6fc80fe08885edba34c166da09465f6/resources/pts_in_hull.npy?raw=true
curl -LJO https://github.com/richzhang/colorization/blob/a1642d6ac6fc80fe08885edba34c166da09465f6/resources/pts_in_hull.npy?raw=true

wget --no-check-certificate --content-disposition http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel
curl -LJO http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel

rm 'pts_in_hull.npy?raw=true'

cd ..
mv model flaskr/Colorizer/
