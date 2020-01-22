# Semantic segmentation 
Notebook for Kaggle competition 'Airbus Ship Detection Challenge' https://www.kaggle.com/c/airbus-ship-detection/overview
Provide Jupyter Notebook of my solution for this problem with all outputs. Also script(not tested)
Notebook was run in Kaggle karnel because of fre GPU. So datasets wasn't instal localy. 29 GB is too large.
### Tools
In project used python with such librarys:
	*jupyter notebook for local usage
	*NumPy for linear algebra
	*Pandas for datasets
	*Matplotlib for plotting and visualization
	*Keras as backend
	*OpenCV for some image processing. Easily replaced by skimage
	*skimage for encodeeng the results for format appropriate to Kaggle submission
	*tqdm for trackbars
### Preprocessing
To normalize ratio of images with and without ships most of images was deleted. Also one problem image was deleted. As a result 80000 images used to train model
### Neural network
Was used U-Net architecture for network. In model exist such layers:
	*Convolution with different number of filters
	*Normalization
	*ReluActivation
	*Pooling
	*Dropout to prevent overfit
First 6 layers reduce image to get different features, next 6 - to return to original size. Dice score uses. Threshold was choosen by using predictions on validation set.
### Results
Unfortunately in real data model do bad job. Despite good accurace and Dice score while teaching in all tested images model get empty masks. As a real work model it cannot be used but it may be good bacelsne and template for future research.
"# Semantic_Segmentation_Kaggle" 
