##  DeceptionNet MNIST

### Files and their meanings
- **check.py**: check code for model architecture--typically needs a checkerboard image going by the name of checker.png to run this.
- **checker.png**
- **demo.png**: Consists of a warped image sample from the DeceptionNet warping module.
- **demo_orig.png**: Consists of the original version of the warped image described above.
- **model.py**: Contains the model architecture:
  - ***class classifier***: Classifier architecture.
  - ***class Conv_block***: Convolutional Block, described as CB in paper.
  - ***class Conv_block_T***: transposed convolutional version of CB
  - ***class decoder_ds***: Distorion module of decoder.
  - ***class deform***: assistant module for decoder_ds.
  - ***class decoder_bg***: Background module of decoder.
  - ***class decoder_ns***: Noise Addition module of decoder.
  - ***class decoder_general***: Assistant module for decoder_ds and decoder_bg
  - ***class encoder***: Encoder phase of architecture
  - ***class encode_mnist***: master module controlling all above modules
- **mnist_basic.py**: Trains the classifier network--basic classifier training loop
  - ***Output***: classifier_basic.pt
- **mnist_warp.py**: Warps the input (using encode_mnist) to bring down accuracy of classification module. Classifier weights frozen
  - ***Output***: classifier_advanced.pt
- **mnist_final.py**: Trains the classifier network on data generated from warping module as well as original data. Warping module weights (encode_mnist) frozen.
  - ***Output***: classifier_compelte.pt
- **test_m.py**: Tests the model weights of the classifier stored in `classifier_complete.pt` against MNIST-M data.

### How to run the files
First run **mnist_basic.py**, followed by **mnist_warp.py** and then **mnist_final.py**. This will make the model learn from the source domain and give classification results for the source domain. Then running on **test_m.py** gives the classification results in the target dataset [MNIST-M]. 
  
