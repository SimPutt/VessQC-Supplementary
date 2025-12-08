# Uncertainty Map Generation 

This GitHub contain all the information and steps related to the pixelwise uncertainty map calculation based on a probabilistic model predictions.

---

## In order to use 

* If you haven't installed the [mmv_im2im](https://github.com/MMV-Lab/mmv_im2im) project, follow the instruction in the related repo to install the project, this is required for the prediction and uncertainity map generation.

* With the project installed clone the actual repo:

```bash
git clone https://github.com/uncerrtainty_repo.git
cd ./ Uncertainty_map 
```


### 🚨 Note: 
Before run the ```run_im2im``` instruction be sure that this run is within the same anaconda enviroment where the [mmv_im2im](https://github.com/MMV-Lab/mmv_im2im) project was installed:

```bash
conda activate im2im_enviroment_name
```

* To run the example just run:

```bash
run_im2im --config path/to/the/map_generation.yaml
```


## Running the code in your own data:

* Follow the coments in the [map_generation.yaml](example.yaml) configuration file according to your data options and save the changes.

Then just run:

```bash
run_im2im --config path/to/the/map_generation.yaml
```
### 🚨 Notes: 
In the [map_generation.yaml](map_generation.yaml) configuration file we provide some help comments in order to undertand and set the uncertainty map generation parameters for a detailed explanation about the calculations and the parameters we provide the [manual](manual.md) file with the full deatils.

A [pretarined](pretrain) Probabilistic Unet  weights are provided for the vessel segmentation example.
