# CompVis
 Project for Computational Visual Perception course at FAU

## Dataset
ImageNet-100

Data file Structure:

    data1

        --class1

            --*.jpeg

            --*.jpeg

        --class2

            --*.jpeg

            --*.jpeg
            
        ...

*_demo files are for debugging

dataloader.ipynb and data.py are for final submission

data_demo.py and data.py are exactly same, except data_demo passes the original image as well as transformed image. So dataloader_demo.ipynb can show an example of how the transformation looks like compared to original

dataloader.ipynb and dataloader.py are the same scripts. They only print out the average loading time for 100 images in each age category and display the different loading times as a graph

dataloader_demo.ipynb is same as dataloader.ipynb (with a smaller batch_size), but it also displays example image pairs for comparison of transforms


Current issues:

    ● dataloader.py has path issues and causes runtime errors, but same script in dataloader.ipynb works fine