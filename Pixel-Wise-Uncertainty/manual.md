# Uncertainty configuration guide

### inference_input: dir (String)

Path to the folder with the images for maps generation.

### inference_output: path (String)

Path to the folder for the output data.

### uncertainity_map (Boolean)

Indicate when to generate the model segmentation and the uncertainity map (True) or just the segmentation (False).

### piexl_dim (Float List)

Indicate the physical pixel dimension taked into accound for the segmentation postprocessing, $[1,1,1]$ is set as default.

### n_class_correction (Int)

The number of classes taked into account for the segmentation postprocessing

### remove_object_size (Float List)

Removes small, isolated regions in the segmentation output. It requires one or more values list representing the minimum volume in $\mu m^3$ an object must have to be kept. Objects smaller than this threshold will be deleted.

### hole_size_threshold (Float List)

Fixes small errors by filling in holes within segmented objects. It requires one or more values representing the maximum area in $\mu m^2$ a hole must have to be filled. Holes larger than this threshold are considered genuine features and will remain open.

### min_thickness_list (Float List)

Applies topology-preserving thinning to the segmented objects. Its purpose is to reduce the thickness of the objects, to a minimal representation in $\mu m$ while ensuring that the connectivity and general shape of the original object are maintained.

### 🚨 Postprocessing Note:

The postprocessing not reflect any change in the uncertainity map just modify the segmentations given for the model.

### n_samples (Int)

Number of samples (Model predictions)  taken into account for the uncertainty map computation.

### multi_pred_mode (String)

Indicate how to compute the final model prediction given $N$ samples, the available options are implemented as follow:

Given a image $I$ and a trained model $M$ we can compute $N$ predictions $\displaystyle \left\{ M_{i}(I) \right\}_{i=1}^{N}$ where each prediction represent one sample for the distribution learned for the probabilistic model.

The final prediction options are:

#### 'single' 

This option just take the regular model logits to generate the prediction, in this case the first computed  $ Pred = M_{1}(I)$

#### 'mean'

Take the mean pixel over the sample logits to generate the final prediction $\displaystyle Pred_{(j,k)} = \frac{1}{N}\sum_{i=1}^{N} M_{i}(I)_{(j,k)}$

#### 'var'

Take the variance pixel over the sample logits to generate the final prediction $\displaystyle Pred_{(j,k)} = \frac{1}{N}\sum_{i=1}^{N} (M_{i}(I)_{(j,k)}-\mu_{i})^2$

### pertubations (String List)

List of pertubation randomly applied to the original image before prediction generation and map computation in aims of generate enough variance between the samples trought a pixelwise Monte Carlo Dropout process.

Given a list of perturbations transformation $\displaystyle \left\{T_t\right\}_{t=1}^N$ for every sample except for the first one (we keept the firts one unchange) we take a random number $t'$ of the available transformations and apply them to the image $I$ before prediction,
given the perturbed image $I'$ and with this a set of perturbed logits predictions.

The available perturbation tranformations are:

#### 'gauss_noise' , 'impulse_noise' , 'speckle_noise' , 'color_jitter' , 'shift' , 'rotation' , 'pixel_dropout'

### compute_mode (String)

Computation method for the uncertainty maps, the available options are:

#### 'variance'

This method compute the variance among the perturbed logits  $\displaystyle Umap_{(j,k)} = \frac{1}{N}\sum_{i=1}^{N} (M_{i}(I')_{(j,k)}-\mu_{i})^2$ this method is useful for segmentations with multiple classes given a map per class, that's why the parameter  'var_reductor' is included.

### var_reductor (Boolean)

If is set True generate a general uncertainty map just taking the minimun variance class in other case it generate one uncertainty map per class present in the segementation.

#### 'prob_inv'

For each class present on the segmentation take the maxima logit per class , then compute the mean complementary probabiliy  $\displaystyle Umap_{(j,k)} = \frac{1}{N}\sum_{i=1}^{N} 1- \max M_{i}(I')_{(j,k)}$

#### 'entropy'

This method compute the total entropy $\displaystyle H[E[P(y|x)]] = -\sum_{c=1}^{C} E[P(y=c|x)] \log(E[P(y=c|x)])$ 

capturing the total uncertainty (epistemic + heuristic) that is the uncertainty related to the presence of noise, quality of the data, device errors , etc, plus the uncertainty related to the model learning.

#### 'mutual_inf'

This method computes the mutual information or bayesian entropy $\displaystyle \text{MI} = H[E[P(y|x)]] - E[H[P(y|x, w)]]$ where $\displaystyle E[H[P(y|x, w)]] = E_{w \sim q(w|D)}[H[P(y|x, w)]] = \frac{1}{N}\sum_{i=1}^{N} H[P(y|x, w_i)]$

capturing the heuristic uncertainty the related one to the model learning.

This computation offer theorical result in the interval $\displaystyle [0,\log(C)]$ where $C$ is the number of clases present in the segmentation due normalizing option we incluthe the parameter 'relative_MI' 

### relative_MI (Boolean)

If is set True the normalization $\displaystyle \text{MI}_{\text{normalized}} = \frac{\text{MI}}{\log(C)}$ it's applied in other case the theorical range $\displaystyle [0,\log(C)]$ is used.


## Other parameters

We include some processing parameters for the probabilistic maps that can be helpful.


### estabilizer (Boolean)

If is set True it prevent the use of very small probabilistic values using a sqrt root monotonic transformation $\displaystyle \sqrt{Umap}$. 

### trunc (Int)

Number of digits to take into accoun after the float point for the  $\displaystyle Umap$, helpful to prevent unecesary large numbers.

### threshold (Float)

Threshold for the uncertainty map  turn into  0 every value under de threshold   $\displaystyle Umap \leq  threshold = 0$ 

### threshold border_correction (Int List )

Number of pixels $[x,y]$ (top/dowm,left/right) to ignore in order to prevent errors at the boundary.
