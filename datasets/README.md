# SyDog-Video: A Synthetic Dog Video Dataset for Temporal Pose Estimation 

## Description
Each verion of the dataset includes 500 synthetic dog videos of 175 frames (87,500 frames) including 2D ground truth such as bounding box coordinates, 33 keypoint labels and segmentation maps. There are 6 versions of the datasets: clean_plate, w_assets, w_assetsPlusPeople, w_people, wo_fur_clean_plate, wo_groundplane. For example, the clean_plate-version of the dataset includes images with HDRI and a ground geometry which represents the floor. There are'nt any distractors such as 3D assets or people in the background present.

## Folder structure
- [clean_plate, ..., wo_groundplane]/
These folders contain the data of the videos e.g. RGB data and labels. The subfolders refer to sequences associated to the type of dog e.g. dog2, labrador, etc. Because the dataset was generated using the Unity Perception package the subfolder (e.g. dog2) contains the following structure including the keypoint labels (Dataset[ID]), RGB data (RGB[ID]) and semantic segmenation information (SemanticSegmentation[ID]). These labels were processings using code from the following link and transformed into JSON annotation files. For the JSON annotation files please refer to the vid_annotations folder.

- dog2/
    - Dataset[ID]
    - Logs
    - RGB[ID]
    - SemanticSegmentation[ID]
- labrador/
    - ...
- pug/
    - ...
- pitbull/
    - ...
- wolf/
    - ...
- vid_annotations/
    The folder contains the annotations for each video sequence e.g. video-000-clean_plate-dog2.json
    The data format of the JSON files are the following: video-[videoID]-[dataset_type]-[dog_type].json
- split_annotations/within_dataset/
    The dataset training/test split files e.g. {[dataset_type]}.json files, which represent the sequences used.
- scripts
	This folder contains a script how to access the data for training or just the individual video sequences.

## Citation
@article{Shooter_SyDogVideo2023,
	abstract = {We aim to estimate the pose of dogs from videos using a temporal deep learning model as this can result in more accurate pose predictions when temporary occlusions or substantial movements occur. Generally, deep learning models require a lot of data to perform well. To our knowledge, public pose datasets containing videos of dogs are non existent. To solve this problem, and avoid manually labelling videos as it can take a lot of time, we generated a synthetic dataset containing 500 videos of dogs performing different actions using Unity3D. Diversity is achieved by randomising parameters such as lighting, backgrounds, camera parameters and the dog's appearance and pose. We evaluate the quality of our synthetic dataset by assessing the model's capacity to generalise to real data. Usually, networks trained on synthetic data perform poorly when evaluated on real data, this is due to the domain gap. As there was still a domain gap after improving the quality of the synthetic dataset and inserting diversity, we bridged the domain gap by applying 2 different methods: fine-tuning and using a mixed dataset to train the network. Additionally, we compare the model pre-trained on synthetic data with models pre-trained on a real-world animal pose datasets. We demonstrate that using the synthetic dataset is beneficial for training models with (small) real-world datasets. Furthermore, we show that pre-training the model with the synthetic dataset is the go to choice rather than pre-training on real-world datasets for solving the pose estimation task from videos of dogs.},
	author = {Shooter, Moira and Malleson, Charles and Hilton, Adrian},
	date = {2023/12/29},
	date-added = {2024-05-23 14:51:34 +0100},
	date-modified = {2024-05-23 14:51:34 +0100},
	doi = {10.1007/s11263-023-01946-z},
	id = {Shooter2023},
	isbn = {1573-1405},
	journal = {International Journal of Computer Vision},
	title = {SyDog-Video: A Synthetic Dog Video Dataset for Temporal Pose Estimation},
	url = {https://doi.org/10.1007/s11263-023-01946-z},
	year = {2023},
	bdsk-url-1 = {https://doi.org/10.1007/s11263-023-01946-z}}


## License
SyDog-Video Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.
