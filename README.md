# CCLGAN

Adversarial network for unsupervised infrared image colorization based on full-scale feature fusion and cosine contrastive learning



This repository is implementation of the ["Adversarial network for unsupervised infrared image colorization based on full-scale feature fusion and cosine contrastive learning"](CCLGAN)by PyTorch.

Dataset Preparation
Please ensure that your datasets are properly downloaded and placed in the datasets folder. This project evaluates the method on three datasets: KAIST, FLIR, and NIR.


Training and Testing
After setting the options and dataset paths, you can directly run the training or testing process.

To view training results and loss plots, you can optionally run python -m visdom.server and click the URL http://localhost:8097.

Training CCLGAN
The checkpoints and intermediate results will be stored at ./checkpoints/KAIST_CCLGAN/web.

Testing CCLGAN
The test results will be saved to an HTML file located at: ./results/KAIST_KAIST_CCLGAN/latest_train/index.html (or test_latest depending on the phase setting).


./datasets/

  ├── KAIST/
  │     ├── trainA, trainB, testA, testB
  ├── FLIR/
  │     ├── trainA, trainB, testA, testB
  └── NIR/
        ├── trainA, trainB, testA, testB


Requirement

**python 3.9, Pytorch=2.0.1, cuda 12.1, RTX 3090 GPU**

**Please refer to the requirements.txt file for details.**

    pip install requirements.txt


After setting the options and dataset paths, you can directly run the training or testing process.

# python3 train.py --dataroot ./datasets/KAIST --name KAIST_CCLGAN --CUT_mode CUT --gpu_ids 0 

# python3 test.py --dataroot ./datasets/KAIST --name KAIST_KAIST_CCLGAN --CUT_mode CUT --gpu_ids 0


Citation

If you use this code for your research, please cite the paper.


Acknowledgments

Our code is developed based on Contrastive Learning for Unpaired Image-to-Image Translation (CUT), https://github.com/taesungp/contrastive-unpaired-translation. We thank the authors for their excellent work and for providing the foundational framework.

