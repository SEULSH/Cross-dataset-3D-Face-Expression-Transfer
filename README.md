# Cross-dataset-3D-Face-Expression-Transfer
The code of our paper named “Cross-dataset 3D face expression transfer via UV representation and autoencoder network.”

Here in the code, for a example, we train and test model using COMA dataset.

The public dataset can be obtained in the website:

https://coma.is.tue.mpg.de/

other used datasets in our paper are also public and available in corresponding papers.

The files including data_process.py, unwrap_code.py, coma_cropped_index.npy are used to get UV maps from mesh shapes in COMA.  To run this code, you need to install packages including psbody, face3d following possible websites:

https://github.com/anuragranj/coma

https://github.com/yfeng95/face3d

For COMA dataset, before getting UV maps, we also remove eye part for excluding influence, which is easy to do using mesh processing tools.

For getting GT shapes, you can train pretrained model by following the original work:  

https://github.com/Gaozhongpai/LSAConvMesh

Then, during the training, you can use the pretrained model to get GT mesh shapes by using latent operations between neutral meshes and expression meshes, which are included in codes of data.py file.
 
In train and test files, you can change data root and code blocking comments to train or test in other datasets and use other functions.

We will keep updating the project. If you have any question, please contact us. Email: 1515646589@qq.com
