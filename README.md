# CompVis
 Project for Computational Visual Perception course at FAU

## Part 1-Implementation of age-dependent infant color sensitivity

### Paper description

During the experiment carried out in the paper "Systematic Measurement of Human Neonatal Color Vision" by RUSSELL J. et. al., 1993, the authors take infant subjects of various ages, and present them with pairs of color cards. One with a color of a certain wavelength and other with achromatic(gray) color of equivalent luminosity.

If less than 50% of infants of age x (in months) are able to differentiate between the two, the age category is branded as insensitive to that color. If more than 50% can find the difference, that age category is considered sensitive to that color. If the proportion is around 50%, the results are considered inconclusive and the sensitivity is considered transitive.

### Algorithmic logic

To implement similar perception algorithmically, we first map the color wavelengths used in the experiment to a corresponding hue angle on HSL color wheel. So blue (475 nm) is mapped to 225 degrees, green (520 nm) is mapped to 135 degrees, yellow (580 nm) is mapped to 60 degrees, red (660 nm) is mapped to 330 degrees.

Now a symmetrical region around these particular hue angles is considered as the region of influence of the corresponding color. 180 to 270 degrees of hue region is considered bluish region. If the infant of age x (in months) is insensitive to blue color, it will have least sensitivity(zero) towards blue color at the center of bluish region, i.e., at 225 degrees the color will appear greyish to the infant. And the sensitivity gradually increases to full capacity at the edges of bluish region, i.e., the colors at edges of blue region are perceived as they are. Same goes for all other colors. If the infant is sensitive to red color, the infant will have equally full sensitivity for all hue angles in the entire reddish region, 270 to 30 degrees. If the infant sensitivity to green color is transitive, it will have full sensitivity for any hue angle within greenish region, if the saturation is above 0.5 on a 0-1 scale. Otherwise, the sensitivity will be lower, based on how far the hue angle is from the center (135 degrees) of greenish region, just like the above example of blue insensitivity in bluish region, i.e, completely greyish at 135 degrees, and unchanged color at the edges of greenish region.

Now the decrease in sensitivity is simulated by decreasing the saturation. The saturation_decrease_factor is set to 0 at the center of the region of color, that the infant is insensitive or transitively sensitive to, and is set to 1 at both the edges of the region. For all other hue angles between the center and edges, the saturation_decrease_factor follows a linear mapping that lies between 0 and 1 respectively. In case of colors where infant has full sensitivity, the saturation_decrease_factor is set to 1 for all hue angles in the color region.

During batch processing, the dataloader takes one image from the dataset, then converts all RGB values to corresponding HSL value at pixel-level. For each pixel, based on hue and saturation, corresponding saturation_decrease_factor is calculated, and multiplied with saturation value. The saturation value thus changes while hue and luminosity remain unchanged. Then these new HSL values are converted back to RGB values for each pixel.

We run this dataloader for four batches of 100 images each, for each age category from 0 to 3 months (0 months being a newborn and 3 months being the age of complete sensitivity for all colors, according to the paper). We thus get, the average runtime for the dataloader for each age category with limited sensitivity and can compare it with the runtime for age of 3 months, with no change in image perception required, hence no transformation of image required.

Since the authors of the paper did not test the color perception in magenta range in the experiment, we divided the magenta region between reddish and bluish region with corresponding sensitivity.

The resulting image transformations and runtimes are presented in images in dataloader_demo.ipynb file.

### Dataset
ImageNet-100

### Data file Structure:

    data1

        --class1

            --*.jpeg

            --*.jpeg

        --class2

            --*.jpeg

            --*.jpeg
            
        ...

### Miscellaneous

*_demo files are for debugging

dataloader.ipynb and data.py are for final submission

data_demo.py and data.py are exactly same, except data_demo passes the original image as well as transformed image. So dataloader_demo.ipynb can show an example of how the transformation looks like compared to original

dataloader.ipynb and dataloader.py are the same scripts. They only print out the average loading time for 100 images in each age category and display the different loading times as a graph

dataloader_demo.ipynb is same as dataloader.ipynb (with a smaller batch_size), but it also displays example image pairs for comparison of transforms


### Current issues:

    ● dataloader.py has path issues and causes runtime errors, but same script in dataloader.ipynb works fine