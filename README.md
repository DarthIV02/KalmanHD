# Research_Final_2

Organization of the files:
- Each file titled Comparison_# is the comparison between DNN, VAE and RegHD using diffrent datasets. They import the models from the file with the respective names. The ones the don't have norm take as input the normal dataset, while the ones that have _Norm, take the normlaized dataset
- DNN, VAE and RegHD are the general code for each model
- Encode: Are RegHD tests with diffrent test for 3 diffrent encodings: Permutation for time series, sin*cos and just sin
- Datasetmine2: Is the dataset loader for the 6 diffrent datasets
- Multi_model_RegHD: Is the implementation of multimodel RegHD
- Multi_modelRegHD_debug: Is diffrent test on Multimodel RegHD, especifically to test the distribution of the clusters
- Multi_Model_RegHD_allterations: Last version of RegHD were I'm testing the diffrent possible alterations of the model t make it more efficient.
- Clustering Tests: Initial test of implementation of each clusterization method (Kmeans + Spectral Clustering) + Visualization of hd vectors on each clusterization
- Mod_Clustering_RegHD: RSME of Normal RegHD VS Kmeans RegHD VS Spectral Clustering RegHD
- Contextual_Aware_Encoding_test: Initial trials for Contextual Aware Encoding
