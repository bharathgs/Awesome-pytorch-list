Awesome-Pytorch-list
========================

![pytorch-logo-dark](https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png)

<p align="center">
	<img src="https://img.shields.io/badge/stars-12400+-brightgreen.svg?style=flat"/>
	<img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat">
</p>

## Contents
- [Pytorch & related libraries](#pytorch--related-libraries)
  - [NLP & Speech Processing](#nlp--Speech-Processing)
  - [Computer Vision](#cv)
  - [Probabilistic/Generative Libraries](#probabilisticgenerative-libraries)
  - [Other libraries](#other-libraries)
- [Tutorials, books & examples](#tutorials-books--examples)
- [Paper implementations](#paper-implementations)
- [Talks & Conferences](#talks--conferences)
- [Pytorch elsewhere](#pytorch-elsewhere)

## Pytorch & related libraries

1. [pytorch](http://pytorch.org): Tensors and Dynamic neural networks in Python with strong GPU acceleration.
2. [Captum](https://github.com/pytorch/captum): Model interpretability and understanding for PyTorch.

### NLP & Speech Processing:

1. [pytorch text](https://github.com/pytorch/text): Torch text related contents.  
2. [pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq): A framework for sequence-to-sequence (seq2seq) models implemented in PyTorch.  
3. [anuvada](https://github.com/Sandeep42/anuvada): Interpretable Models for NLP using PyTorch.
4. [audio](https://github.com/pytorch/audio): simple audio I/O for pytorch.
5. [loop](https://github.com/facebookresearch/loop): A method to generate speech across multiple speakers
6. [fairseq-py](https://github.com/facebookresearch/fairseq-py): Facebook AI Research Sequence-to-Sequence Toolkit written in Python.
7. [speech](https://github.com/awni/speech): PyTorch ASR Implementation.
8. [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py): Open-Source Neural Machine Translation in PyTorch http://opennmt.net 
9. [neuralcoref](https://github.com/huggingface/neuralcoref): State-of-the-art coreference resolution based on neural nets and spaCy huggingface.co/coref
10. [sentiment-discovery](https://github.com/NVIDIA/sentiment-discovery): Unsupervised Language Modeling at scale for robust sentiment classification.
11. [MUSE](https://github.com/facebookresearch/MUSE): A library for Multilingual Unsupervised or Supervised word Embeddings
12. [nmtpytorch](https://github.com/lium-lst/nmtpytorch): Neural Machine Translation Framework in PyTorch.
13. [pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet): An implementation of WaveNet with fast generation
14. [Tacotron-pytorch](https://github.com/soobinseo/Tacotron-pytorch): Tacotron: Towards End-to-End Speech Synthesis.
15. [AllenNLP](https://github.com/allenai/allennlp): An open-source NLP research library, built on PyTorch.
16. [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP): Text utilities and datasets for PyTorch pytorchnlp.readthedocs.io
17. [quick-nlp](https://github.com/outcastofmusic/quick-nlp): Pytorch NLP library based on FastAI. 
18. [TTS](https://github.com/mozilla/TTS): Deep learning for Text2Speech
19. [LASER](https://github.com/facebookresearch/LASER): Language-Agnostic SEntence Representations
20. [pyannote-audio](https://github.com/pyannote/pyannote-audio): Neural building blocks for speaker diarization: speech activity detection, speaker change detection, speaker embedding
21. [gensen](https://github.com/Maluuba/gensen): Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning.
22. [translate](https://github.com/pytorch/translate): Translate - a PyTorch Language Library.
23. [espnet](https://github.com/espnet/espnet): End-to-End Speech Processing Toolkit espnet.github.io/espnet
24. [pythia](https://github.com/facebookresearch/pythia): A software suite for Visual Question Answering
25. [UnsupervisedMT](https://github.com/facebookresearch/UnsupervisedMT): Phrase-Based & Neural Unsupervised Machine Translation.
26. [jiant](https://github.com/jsalt18-sentence-repl/jiant): The jiant sentence representation learning toolkit. 
27. [BERT-PyTorch](https://github.com/codertimo/BERT-pytorch): Pytorch implementation of Google AI's 2018 BERT, with simple annotation
28. [InferSent](https://github.com/facebookresearch/InferSent): Sentence embeddings (InferSent) and training code for NLI.
29. [uis-rnn](https://github.com/google/uis-rnn):This is the library for the Unbounded Interleaved-State Recurrent Neural Network (UIS-RNN) algorithm, corresponding to the paper Fully Supervised Speaker Diarization. arxiv.org/abs/1810.04719 
30. [flair](https://github.com/zalandoresearch/flair): A very simple framework for state-of-the-art Natural Language Processing (NLP)
31. [pytext](https://github.com/facebookresearch/pytext): A natural language modeling framework based on PyTorch fb.me/pytextdocs
32. [voicefilter](https://github.com/mindslab-ai/voicefilter): Unofficial PyTorch implementation of Google AI's VoiceFilter system http://swpark.me/voicefilter
33. [BERT-NER](https://github.com/kamalkraj/BERT-NER): Pytorch-Named-Entity-Recognition-with-BERT. 
34. [transfer-nlp](https://github.com/feedly/transfer-nlp): NLP library designed for flexible research and development
35. [texar-pytorch](https://github.com/asyml/texar-pytorch): Toolkit for Machine Learning and Text Generation, in PyTorch texar.io
36. [pytorch-kaldi](https://github.com/mravanelli/pytorch-kaldi): pytorch-kaldi is a project for developing state-of-the-art DNN/RNN hybrid speech recognition systems. The DNN part is managed by pytorch, while feature extraction, label computation, and decoding are performed with the kaldi toolkit.
37. [NeMo](https://github.com/NVIDIA/NeMo): Neural Modules: a toolkit for conversational AI nvidia.github.io/NeMo
38. [pytorch-struct](https://github.com/harvardnlp/pytorch-struct): A library of vectorized implementations of core structured prediction algorithms (HMM, Dep Trees, CKY, ..,)
39. [espresso](https://github.com/freewym/espresso): Espresso: A Fast End-to-End Neural Speech Recognition Toolkit
40. [transformers](https://github.com/huggingface/transformers): huggingface Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch. huggingface.co/transformers
41. [reformer-pytorch](https://github.com/lucidrains/reformer-pytorch): Reformer, the efficient Transformer, in Pytorch
42. [torch-metrics](https://github.com/enochkan/torch-metrics): Metrics for model evaluation in pytorch
43. [speechbrain](https://github.com/speechbrain/speechbrain): SpeechBrain is an open-source and all-in-one speech toolkit based on PyTorch.
44. [Backprop](https://github.com/backprop-ai/backprop): Backprop makes it simple to use, finetune, and deploy state-of-the-art ML models.

### CV:

1. [pytorch vision](https://github.com/pytorch/vision): Datasets, Transforms and Models specific to Computer Vision.
2. [pt-styletransfer](https://github.com/tymokvo/pt-styletransfer): Neural style transfer as a class in PyTorch.
3. [OpenFacePytorch](https://github.com/thnkim/OpenFacePytorch):  PyTorch module to use OpenFace's nn4.small2.v1.t7 model
4. [img_classification_pk_pytorch](https://github.com/felixgwu/img_classification_pk_pytorch): Quickly comparing your image classification models with the state-of-the-art models (such as DenseNet, ResNet, ...)
5. [SparseConvNet](https://github.com/facebookresearch/SparseConvNet): Submanifold sparse convolutional networks.
6. [Convolution_LSTM_pytorch](https://github.com/automan000/Convolution_LSTM_pytorch): A multi-layer convolution LSTM module
7. [face-alignment](https://github.com/1adrianb/face-alignment): :fire: 2D and 3D Face alignment library build using pytorch adrianbulat.com
8. [pytorch-semantic-segmentation](https://github.com/ZijunDeng/pytorch-semantic-segmentation): PyTorch for Semantic Segmentation.
9. [RoIAlign.pytorch](https://github.com/longcw/RoIAlign.pytorch): This is a PyTorch version of RoIAlign. This implementation is based on crop_and_resize and supports both forward and backward on CPU and GPU.
10. [pytorch-cnn-finetune](https://github.com/creafz/pytorch-cnn-finetune): Fine-tune pretrained Convolutional Neural Networks with PyTorch.
11. [detectorch](https://github.com/ignacio-rocco/detectorch): Detectorch - detectron for PyTorch
12. [Augmentor](https://github.com/mdbloice/Augmentor): Image augmentation library in Python for machine learning. http://augmentor.readthedocs.io
13. [s2cnn](https://github.com/jonas-koehler/s2cnn): 
This library contains a PyTorch implementation of the SO(3) equivariant CNNs for spherical signals (e.g. omnidirectional cameras, signals on the globe)
14. [TorchCV](https://github.com/donnyyou/torchcv): A PyTorch-Based Framework for Deep Learning in Computer Vision. 
15. [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark): Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch.
16. [image-classification-mobile](https://github.com/osmr/imgclsmob): Collection of classification models pretrained on the ImageNet-1K.
17. [medicaltorch](https://github.com/perone/medicaltorch): A medical imaging framework for Pytorch http://medicaltorch.readthedocs.io
18. [albumentations](https://github.com/albu/albumentations): Fast image augmentation library.
19. [kornia](https://github.com/arraiyopensource/kornia): Differentiable computer vision library.
20. [pytorch-text-recognition](https://github.com/s3nh/pytorch-text-recognition): Text recognition combo - CRAFT + CRNN.
21. [facenet-pytorch](https://github.com/timesler/facenet-pytorch): Pretrained Pytorch face detection and recognition models ported from davidsandberg/facenet.
22. [detectron2](https://github.com/facebookresearch/detectron2): Detectron2 is FAIR's next-generation research platform for object detection and segmentation.
23. [vedaseg](https://github.com/Media-Smart/vedaseg): A semantic segmentation framework by pyotrch
24. [ClassyVision](https://github.com/facebookresearch/ClassyVision): An end-to-end PyTorch framework for image and video classification.
25. [detecto](https://github.com/alankbi/detecto):Computer vision in Python with less than 10 lines of code
26. [pytorch3d](https://github.com/facebookresearch/pytorch3d): PyTorch3D is FAIR's library of reusable components for deep learning with 3D data pytorch3d.org
27. [MMDetection](https://github.com/open-mmlab/mmdetection): MMDetection is an open source object detection toolbox, a part of the [OpenMMLab project](https://open-mmlab.github.io/).
28. [neural-dream](https://github.com/ProGamerGov/neural-dream): A PyTorch implementation of the DeepDream algorithm. Creates dream-like hallucinogenic visuals.
29. [FlashTorch](https://github.com/MisaOgura/flashtorch): Visualization toolkit for neural networks in PyTorch!
30. [Lucent](https://github.com/greentfrapp/lucent): Tensorflow and OpenAI Clarity's Lucid adapted for PyTorch.
31. [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): MMDetection3D is OpenMMLab's next-generation platform for general 3D object detection, a part of the [OpenMMLab project](https://open-mmlab.github.io/).
32. [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): MMSegmentation is a semantic segmentation toolbox and benchmark, a part of the [OpenMMLab project](https://open-mmlab.github.io/).
33. [MMEditing](https://github.com/open-mmlab/mmediting): MMEditing is a image and video editing toolbox, a part of the [OpenMMLab project](https://open-mmlab.github.io/).
34. [MMAction2](https://github.com/open-mmlab/mmaction2): MMAction2 is OpenMMLab's next generation action understanding toolbox and benchmark, a part of the [OpenMMLab project](https://open-mmlab.github.io/).
35. [MMPose](https://github.com/open-mmlab/mmpose): MMPose is a pose estimation toolbox and benchmark, a part of the [OpenMMLab project](https://open-mmlab.github.io/).
36. [lightly](https://github.com/lightly-ai/lightly) - Lightly is a computer vision framework for self-supervised learning.
37. [RoMa](https://naver.github.io/roma/): a lightweight and efficient library to deal with 3D rotations.


### Probabilistic/Generative Libraries:

1. [ptstat](https://github.com/stepelu/ptstat): Probabilistic Programming and Statistical Inference in PyTorch
2. [pyro](https://github.com/uber/pyro): Deep universal probabilistic programming with Python and PyTorch http://pyro.ai
3. [probtorch](https://github.com/probtorch/probtorch): Probabilistic Torch is library for deep generative models that extends PyTorch.
4. [paysage](https://github.com/drckf/paysage): Unsupervised learning and generative models in python/pytorch.
5. [pyvarinf](https://github.com/ctallec/pyvarinf): Python package facilitating the use of Bayesian Deep Learning methods with Variational Inference for PyTorch. 
6. [pyprob](https://github.com/probprog/pyprob): A PyTorch-based library for probabilistic programming and inference compilation.
7. [mia](https://github.com/spring-epfl/mia): A library for running membership inference attacks against ML models. 
8. [pro_gan_pytorch](https://github.com/akanimax/pro_gan_pytorch): ProGAN package implemented as an extension of PyTorch nn.Module.
9. [botorch](https://github.com/pytorch/botorch): Bayesian optimization in PyTorch

### Other libraries:

1. [pytorch extras](https://github.com/mrdrozdov/pytorch-extras): Some extra features for pytorch.    
2. [functional zoo](https://github.com/szagoruyko/functional-zoo): PyTorch, unlike lua torch, has autograd in it's core, so using modular structure of torch.nn modules is not necessary, one can easily allocate needed Variables and write a function that utilizes them, which is sometimes more convenient. This repo contains model definitions in this functional way, with pretrained weights for some models. 
3. [torch-sampling](https://github.com/ncullen93/torchsample): This package provides a set of transforms and data structures for sampling from in-memory or out-of-memory data. 
4. [torchcraft-py](https://github.com/deepcraft/torchcraft-py): Python wrapper for TorchCraft, a bridge between Torch and StarCraft for AI research.
5. [aorun](https://github.com/ramon-oliveira/aorun): Aorun intend to be a Keras with PyTorch as backend. 
6. [logger](https://github.com/oval-group/logger): A simple logger for experiments.
7. [PyTorch-docset](https://github.com/iamaziz/PyTorch-docset): PyTorch docset! use with Dash, Zeal, Velocity, or LovelyDocs.  
8. [convert_torch_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch): Convert torch t7 model to pytorch model and source.
9. [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch): The goal of this repo is to help to reproduce research papers results.  
10. [pytorch_fft](https://github.com/locuslab/pytorch_fft): PyTorch wrapper for FFTs
11. [caffe_to_torch_to_pytorch](https://github.com/fanq15/caffe_to_torch_to_pytorch)
12. [pytorch-extension](https://github.com/sniklaus/pytorch-extension): This is a CUDA extension for PyTorch which computes the Hadamard product of two tensors.
13. [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch): This module saves PyTorch tensors in tensorboard format for inspection. Currently supports scalar, image, audio, histogram features in tensorboard.
14. [gpytorch](https://github.com/jrg365/gpytorch): GPyTorch is a Gaussian Process library, implemented using PyTorch. It is designed for creating flexible and modular Gaussian Process models with ease, so that you don't have to be an expert to use GPs.
15. [spotlight](https://github.com/maciejkula/spotlight): Deep recommender models using PyTorch.
16. [pytorch-cns](https://github.com/awentzonline/pytorch-cns): Compressed Network Search with PyTorch
17. [pyinn](https://github.com/szagoruyko/pyinn): CuPy fused PyTorch neural networks ops
18. [inferno](https://github.com/nasimrahaman/inferno): A utility library around PyTorch
19. [pytorch-fitmodule](https://github.com/henryre/pytorch-fitmodule): Super simple fit method for PyTorch modules
20. [inferno-sklearn](https://github.com/dnouri/inferno): A scikit-learn compatible neural network library that wraps pytorch.
21. [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert): convert between pytorch, caffe prototxt/weights and darknet cfg/weights
22. [pytorch2caffe](https://github.com/longcw/pytorch2caffe): Convert PyTorch model to Caffemodel
23. [pytorch-tools](https://github.com/nearai/pytorch-tools): Tools for PyTorch
24. [sru](https://github.com/taolei87/sru): Training RNNs as Fast as CNNs (arxiv.org/abs/1709.02755)
25. [torch2coreml](https://github.com/prisma-ai/torch2coreml): Torch7 -> CoreML
26. [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding): PyTorch Deep Texture Encoding Network http://hangzh.com/PyTorch-Encoding
27. [pytorch-ctc](https://github.com/ryanleary/pytorch-ctc): PyTorch-CTC is an implementation of CTC (Connectionist Temporal Classification) beam search decoding for PyTorch. C++ code borrowed liberally from TensorFlow with some improvements to increase flexibility.
28. [candlegp](https://github.com/t-vi/candlegp): Gaussian Processes in Pytorch. 
29. [dpwa](https://github.com/loudinthecloud/dpwa): Distributed Learning by Pair-Wise Averaging. 
30. [dni-pytorch](https://github.com/koz4k/dni-pytorch): Decoupled Neural Interfaces using Synthetic Gradients for PyTorch.
31. [skorch](https://github.com/dnouri/skorch): A scikit-learn compatible neural network library that wraps pytorch
32. [ignite](https://github.com/pytorch/ignite): Ignite is a high-level library to help with training neural networks in PyTorch.
33. [Arnold](https://github.com/glample/Arnold): Arnold - DOOM Agent
34. [pytorch-mcn](https://github.com/albanie/pytorch-mcn): Convert models from MatConvNet to PyTorch
35. [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch): A simplified implemention of Faster R-CNN with competitive performance.
36. [generative_zoo](https://github.com/DL-IT/generative_zoo): generative_zoo is a repository that provides working implementations of some generative models in PyTorch.
37. [pytorchviz](https://github.com/szagoruyko/pytorchviz): A small package to create visualizations of PyTorch execution graphs. 
38. [cogitare](https://github.com/cogitare-ai/cogitare): Cogitare - A Modern, Fast, and Modular Deep Learning and Machine Learning framework in Python. 
39. [pydlt](https://github.com/dmarnerides/pydlt): PyTorch based Deep Learning Toolbox
40. [semi-supervised-pytorch](https://github.com/wohlert/semi-supervised-pytorch): Implementations of different VAE-based semi-supervised and generative models in PyTorch. 
41. [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster): PyTorch Extension Library of Optimised Graph Cluster Algorithms.
42. [neural-assembly-compiler](https://github.com/aditya-khant/neural-assembly-compiler): A neural assembly compiler for pyTorch based on adaptive-neural-compilation. 
43. [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch): Convert Caffe models to PyTorch.
44. [extension-cpp](https://github.com/pytorch/extension-cpp): C++ extensions in PyTorch
45. [pytoune](https://github.com/GRAAL-Research/pytoune): A Keras-like framework and utilities for PyTorch
46. [jetson-reinforcement](https://github.com/dusty-nv/jetson-reinforcement): Deep reinforcement learning libraries for NVIDIA Jetson TX1/TX2 with PyTorch, OpenAI Gym, and Gazebo robotics simulator.
47. [matchbox](https://github.com/salesforce/matchbox): Write PyTorch code at the level of individual examples, then run it efficiently on minibatches.
48. [torch-two-sample](https://github.com/josipd/torch-two-sample): A PyTorch library for two-sample tests
49. [pytorch-summary](https://github.com/sksq96/pytorch-summary): Model summary in PyTorch similar to `model.summary()` in Keras
50. [mpl.pytorch](https://github.com/BelBES/mpl.pytorch): Pytorch implementation of MaxPoolingLoss.
51. [scVI-dev](https://github.com/YosefLab/scVI-dev): Development branch of the scVI project in PyTorch
52. [apex](https://github.com/NVIDIA/apex): An Experimental PyTorch Extension(will be deprecated at a later point)
53. [ELF](https://github.com/pytorch/ELF): ELF: a platform for game research.
54. [Torchlite](https://github.com/EKami/Torchlite): A high level library on top of(not only) Pytorch
55. [joint-vae](https://github.com/Schlumberger/joint-vae): Pytorch implementation of JointVAE, a framework for disentangling continuous and discrete factors of variation star2
56. [SLM-Lab](https://github.com/kengz/SLM-Lab): Modular Deep Reinforcement Learning framework in PyTorch.
57. [bindsnet](https://github.com/Hananel-Hazan/bindsnet): A Python package used for simulating spiking neural networks (SNNs) on CPUs or GPUs using PyTorch
58. [pro_gan_pytorch](https://github.com/akanimax/pro_gan_pytorch): ProGAN package implemented as an extension of PyTorch nn.Module
59. [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric): Geometric Deep Learning Extension Library for PyTorch
60. [torchplus](https://github.com/knighton/torchplus): Implements the + operator on PyTorch modules, returning sequences.
61. [lagom](https://github.com/zuoxingdong/lagom): lagom: A light PyTorch infrastructure to quickly prototype reinforcement learning algorithms.
62. [torchbearer](https://github.com/ecs-vlc/torchbearer): torchbearer: A model training library for researchers using PyTorch.
63. [pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl): Reinforcement Learning with Model-Agnostic Meta-Learning in Pytorch. 
64. [NALU](https://github.com/bharathgs/NALU): Basic pytorch implementation of NAC/NALU from Neural Arithmetic Logic Units paper by trask et.al arxiv.org/pdf/1808.00508.pdf
66. [QuCumber](https://github.com/PIQuIL/QuCumber): Neural Network Many-Body Wavefunction Reconstruction
67. [magnet](https://github.com/MagNet-DL/magnet): Deep Learning Projects that Build Themselves http://magnet-dl.readthedocs.io/
68. [opencv_transforms](https://github.com/jbohnslav/opencv_transforms): OpenCV implementation of Torchvision's image augmentations
69. [fastai](https://github.com/fastai/fastai): The fast.ai deep learning library, lessons, and tutorials
70. [pytorch-dense-correspondence](https://github.com/RobotLocomotion/pytorch-dense-correspondence): Code for "Dense Object Nets: Learning Dense Visual Object Descriptors By and For Robotic Manipulation" arxiv.org/pdf/1806.08756.pdf
71. [colorization-pytorch](https://github.com/richzhang/colorization-pytorch): PyTorch reimplementation of Interactive Deep Colorization richzhang.github.io/ideepcolor
72. [beauty-net](https://github.com/cms-flash/beauty-net): A simple, flexible, and extensible template for PyTorch. It's beautiful.
73. [OpenChem](https://github.com/Mariewelt/OpenChem): OpenChem: Deep Learning toolkit for Computational Chemistry and Drug Design Research mariewelt.github.io/OpenChem 
74. [torchani](https://github.com/aiqm/torchani): Accurate Neural Network Potential on PyTorch aiqm.github.io/torchani
75. [PyTorch-LBFGS](https://github.com/hjmshi/PyTorch-LBFGS): A PyTorch implementation of L-BFGS.
76. [gpytorch](https://github.com/cornellius-gp/gpytorch): A highly efficient and modular implementation of Gaussian Processes in PyTorch.
77. [hessian](https://github.com/mariogeiger/hessian): hessian in pytorch. 
78. [vel](https://github.com/MillionIntegrals/vel): Velocity in deep-learning research.
79. [nonechucks](https://github.com/msamogh/nonechucks): Skip bad items in your PyTorch DataLoader, use Transforms as Filters, and more!
80. [torchstat](https://github.com/Swall0w/torchstat): Model analyzer in PyTorch.
81. [QNNPACK](https://github.com/pytorch/QNNPACK): Quantized Neural Network PACKage - mobile-optimized implementation of quantized neural network operators.
82. [torchdiffeq](https://github.com/rtqichen/torchdiffeq): Differentiable ODE solvers with full GPU support and O(1)-memory backpropagation.
83. [redner](https://github.com/BachiLi/redner): A differentiable Monte Carlo path tracer
84. [pixyz](https://github.com/masa-su/pixyz): a library for developing deep generative models in a more concise, intuitive and extendable way. 
85. [euclidesdb](https://github.com/perone/euclidesdb): A multi-model machine learning feature embedding database http://euclidesdb.readthedocs.io 
86. [pytorch2keras](https://github.com/nerox8664/pytorch2keras): Convert PyTorch dynamic graph to Keras model.
87. [salad](https://github.com/domainadaptation/salad): Semi-Supervised Learning and Domain Adaptation.
88. [netharn](https://github.com/Erotemic/netharn): Parameterized fit and prediction harnesses for pytorch.
89. [dgl](https://github.com/dmlc/dgl): Python package built to ease deep learning on graph, on top of existing DL frameworks. http://dgl.ai. 
90. [gandissect](https://github.com/CSAILVision/gandissect): Pytorch-based tools for visualizing and understanding the neurons of a GAN. gandissect.csail.mit.edu 
91. [delira](https://github.com/justusschock/delira): Lightweight framework for fast prototyping and training deep neural networks in medical imaging delira.rtfd.io
92. [mushroom](https://github.com/AIRLab-POLIMI/mushroom): Python library for Reinforcement Learning experiments.
93. [Xlearn](https://github.com/thuml/Xlearn): Transfer Learning Library
94. [geoopt](https://github.com/ferrine/geoopt): Riemannian Adaptive Optimization Methods with pytorch optim
95. [vegans](https://github.com/unit8co/vegans): A library providing various existing GANs in PyTorch.
96. [torchgeometry](https://github.com/arraiyopensource/torchgeometry): TGM: PyTorch Geometry
97. [AdverTorch](https://github.com/BorealisAI/advertorch): A Toolbox for Adversarial Robustness (attack/defense/training) Research
98. [AdaBound](https://github.com/Luolc/AdaBound): An optimizer that trains as fast as Adam and as good as SGD.a
99. [fenchel-young-losses](https://github.com/mblondel/fenchel-young-losses): Probabilistic classification in PyTorch/TensorFlow/scikit-learn with Fenchel-Young losses
100. [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter): Count the FLOPs of your PyTorch model.
101. [Tor10](https://github.com/kaihsin/Tor10): A Generic Tensor-Network library that is designed for quantum simulation, base on the pytorch.
102. [Catalyst](https://github.com/catalyst-team/catalyst): High-level utils for PyTorch DL & RL research. It was developed with a focus on reproducibility, fast experimentation and code/ideas reusing. Being able to research/develop something new, rather than write another regular train loop.
103. [Ax](https://github.com/facebook/Ax): Adaptive Experimentation Platform
104. [pywick](https://github.com/achaiah/pywick): High-level batteries-included neural network training library for Pytorch
105. [torchgpipe](https://github.com/kakaobrain/torchgpipe): A GPipe implementation in PyTorch torchgpipe.readthedocs.io
106. [hub](https://github.com/pytorch/hub): Pytorch Hub is a pre-trained model repository designed to facilitate research reproducibility.
107. [pytorch-lightning](https://github.com/williamFalcon/pytorch-lightning): Rapid research framework for Pytorch. The researcher's version of keras.
108. [Tor10](https://github.com/kaihsin/Tor10): A Generic Tensor-Network library that is designed for quantum simulation, base on the pytorch.
109. [tensorwatch](https://github.com/microsoft/tensorwatch): Debugging, monitoring and visualization for Deep Learning and Reinforcement Learning from Microsoft Research.
110. [wavetorch](https://github.com/fancompute/wavetorch): Numerically solving and backpropagating through the wave equation arxiv.org/abs/1904.12831
111. [diffdist](https://github.com/ag14774/diffdist): diffdist is a python library for pytorch. It extends the default functionality of torch.autograd and adds support for differentiable communication between processes. 
112. [torchprof](https://github.com/awwong1/torchprof): A minimal dependency library for layer-by-layer profiling of Pytorch models.
113. [osqpth](https://github.com/oxfordcontrol/osqpth): The differentiable OSQP solver layer for PyTorch. 
114. [mctorch](https://github.com/mctorch/mctorch): A manifold optimization library for deep learning. 
115. [pytorch-hessian-eigenthings](https://github.com/noahgolmant/pytorch-hessian-eigenthings): Efficient PyTorch Hessian eigendecomposition using the Hessian-vector product and stochastic power iteration. 
116. [MinkowskiEngine](https://github.com/StanfordVL/MinkowskiEngine): Minkowski Engine is an auto-diff library for generalized sparse convolutions and high-dimensional sparse tensors.
117. [pytorch-cpp-rl](https://github.com/Omegastick/pytorch-cpp-rl): PyTorch C++ Reinforcement Learning
118. [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt): PyTorch extensions for fast R&D prototyping and Kaggle farming
119. [argus-tensor-stream](https://github.com/Fonbet/argus-tensor-stream): A library for real-time video stream decoding to CUDA memory tensorstream.argus-ai.com
120. [macarico](https://github.com/hal3/macarico): learning to search in pytorch
121. [rlpyt](https://github.com/astooke/rlpyt): Reinforcement Learning in PyTorch
122. [pywarm](https://github.com/blue-season/pywarm): A cleaner way to build neural networks for PyTorch. blue-season.github.io/pywarm
123. [learn2learn](https://github.com/learnables/learn2learn): PyTorch Meta-learning Framework for Researchers http://learn2learn.net
124. [torchbeast](https://github.com/facebookresearch/torchbeast): A PyTorch Platform for Distributed RL
125. [higher](https://github.com/facebookresearch/higher): higher is a pytorch library allowing users to obtain higher order gradients over losses spanning training loops rather than individual training steps.
126. [Torchelie](https://github.com/Vermeille/Torchelie/): Torchélie is a set of utility functions, layers, losses, models, trainers and other things for PyTorch. torchelie.readthedocs.org 
127. [CrypTen](https://github.com/facebookresearch/CrypTen): CrypTen is a Privacy Preserving Machine Learning framework written using PyTorch that allows researchers and developers to train models using encrypted data. CrypTen currently supports Secure multi-party computation as its encryption mechanism.
128. [cvxpylayers](https://github.com/cvxgrp/cvxpylayers): cvxpylayers is a Python library for constructing differentiable convex optimization layers in PyTorch
129. [RepDistiller](https://github.com/HobbitLong/RepDistiller): Contrastive Representation Distillation (CRD), and benchmark of recent knowledge distillation methods
130. [kaolin](https://github.com/NVIDIAGameWorks/kaolin): PyTorch library aimed at accelerating 3D deep learning research
131. [PySNN](https://github.com/BasBuller/PySNN): Efficient Spiking Neural Network framework, built on top of PyTorch for GPU acceleration.
132. [sparktorch](https://github.com/dmmiller612/sparktorch): Train and run Pytorch models on Apache Spark.
133. [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning): The easiest way to use metric learning in your application. Modular, flexible, and extensible. Written in PyTorch.
134. [autonomous-learning-library](https://github.com/cpnota/autonomous-learning-library): A PyTorch library for building deep reinforcement learning agents.
135. [flambe](https://github.com/asappresearch/flambe): An ML framework to accelerate research and its path to production. flambe.ai
136. [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer): Collections of modern optimization algorithms for PyTorch, includes: AccSGD, AdaBound, AdaMod, DiffGrad, Lamb, RAdam, RAdam, Yogi.
137. [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE): A Collection of Variational Autoencoders (VAE) in PyTorch.
138. [ray](https://github.com/ray-project/ray): A fast and simple framework for building and running distributed applications. Ray is packaged with RLlib, a scalable reinforcement learning library, and Tune, a scalable hyperparameter tuning library. ray.io
139. [Pytorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal): A temporal extension library for PyTorch Geometric 
140. [Poutyne](https://github.com/GRAAL-Research/poutyne): A Keras-like framework for PyTorch that handles much of the boilerplating code needed to train neural networks.
141. [Pytorch-Toolbox](https://github.com/PistonY/torch-toolbox): This is toolbox project for Pytorch. Aiming to make you write Pytorch code more easier, readable and concise.
142. [Pytorch-contrib](https://github.com/pytorch/contrib): It contains reviewed implementations of ideas from recent machine learning papers.
143. [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch): It contains an op-for-op PyTorch reimplementation of EfficientNet, along with pre-trained models and examples.
144. [PyTorch/XLA](https://github.com/pytorch/xla): PyTorch/XLA is a Python package that uses the XLA deep learning compiler to connect the PyTorch deep learning framework and Cloud TPUs.
145. [webdataset](https://github.com/tmbdev/webdataset): WebDataset is a PyTorch Dataset (IterableDataset) implementation providing efficient access to datasets stored in POSIX tar archives.
146. [volksdep](https://github.com/Media-Smart/volksdep): volksdep is an open-source toolbox for deploying and accelerating PyTorch, Onnx and Tensorflow models with TensorRT.
147. [PyTorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN): StudioGAN is a Pytorch library providing implementations of representative Generative Adversarial Networks (GANs) for conditional/unconditional image generation. StudioGAN aims to offer an identical playground for modern GANs so that machine learning researchers can readily compare and analyze a new idea.
148. [torchdrift](https://github.com/torchdrift/torchdrift/): drift detection library
149. [accelerate](https://github.com/huggingface/accelerate) : A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision 
150. [lightning-transformers](https://github.com/PyTorchLightning/lightning-transformers):  Flexible interface for high-performance research using SOTA Transformers leveraging Pytorch Lightning, Transformers, and Hydra. 
151. [Flower](https://flower.dev/) A unified approach to federated learning, analytics, and evaluation. It allows to federated any machine learning workload.
152. [lightning-flash](https://github.com/PyTorchLightning/lightning-flash): Flash is a collection of tasks for fast prototyping, baselining and fine-tuning scalable Deep Learning models, built on PyTorch Lightning.
153. [Pytorch Geometric Signed Directed](https://github.com/SherylHYX/pytorch_geometric_signed_directed): A signed and directed extension library for PyTorch Geometric. 
154. [Koila](https://github.com/rentruewang/koila): A simple wrapper around pytorch that prevents CUDA out of memory issues.
155. [Renate](https://github.com/awslabs/renate): A library for real-world continual learning.

## Tutorials, books, & examples

1. **[Practical Pytorch](https://github.com/spro/practical-pytorch)**: Tutorials explaining different RNN models
2. [DeepLearningForNLPInPytorch](https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html): An IPython Notebook tutorial on deep learning, with an emphasis on Natural Language Processing. 
3. [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial): tutorial for researchers to learn deep learning with pytorch.
4.  [pytorch-exercises](https://github.com/keon/pytorch-exercises): pytorch-exercises collection. 
5.  [pytorch tutorials](https://github.com/pytorch/tutorials): Various pytorch tutorials. 
6.  [pytorch examples](https://github.com/pytorch/examples):  A repository showcasing examples of using pytorch 
7. [pytorch practice](https://github.com/napsternxg/pytorch-practice): Some example scripts on pytorch.  
8.  [pytorch mini tutorials](https://github.com/vinhkhuc/PyTorch-Mini-Tutorials):  Minimal tutorials for PyTorch adapted from Alec Radford's Theano tutorials. 
9.  [pytorch text classification](https://github.com/xiayandi/Pytorch_text_classification): A simple implementation of CNN based text classification in Pytorch 
10. [cats vs dogs](https://github.com/desimone/pytorch-cat-vs-dogs): Example of network fine-tuning in pytorch for the kaggle competition Dogs vs. Cats Redux: Kernels Edition. Currently #27 (0.05074) on the leaderboard.  
11. [convnet](https://github.com/eladhoffer/convNet.pytorch): This is a complete training example for Deep Convolutional Networks on various datasets (ImageNet, Cifar10, Cifar100, MNIST).
12. [pytorch-generative-adversarial-networks](https://github.com/mailmahee/pytorch-generative-adversarial-networks): simple generative adversarial network (GAN) using PyTorch.   
13. [pytorch containers](https://github.com/amdegroot/pytorch-containers): This repository aims to help former Torchies more seamlessly transition to the "Containerless" world of PyTorch by providing a list of PyTorch implementations of Torch Table Layers.  
14. [T-SNE in pytorch](https://github.com/cemoody/topicsne): t-SNE experiments in pytorch 
15. [AAE_pytorch](https://github.com/fducau/AAE_pytorch): Adversarial Autoencoders (with Pytorch). 
16. [Kind_PyTorch_Tutorial](https://github.com/GunhoChoi/Kind_PyTorch_Tutorial): Kind PyTorch Tutorial for beginners.  
17.  [pytorch-poetry-gen](https://github.com/justdark/pytorch-poetry-gen): a char-RNN based on pytorch.  
18. [pytorch-REINFORCE](https://github.com/JamesChuanggg/pytorch-REINFORCE): PyTorch implementation of REINFORCE, This repo supports both continuous and discrete environments in OpenAI gym.
19.  **[PyTorch-Tutorial](https://github.com/MorvanZhou/PyTorch-Tutorial)**: Build your neural network easy and fast  https://morvanzhou.github.io/tutorials/ 
20. [pytorch-intro](https://github.com/joansj/pytorch-intro): A couple of scripts to illustrate how to do CNNs and RNNs in PyTorch
21. [pytorch-classification](https://github.com/bearpaw/pytorch-classification): A unified framework for the image classification task on CIFAR-10/100 and ImageNet.
22. [pytorch_notebooks - hardmaru](https://github.com/hardmaru/pytorch_notebooks): Random tutorials created in NumPy and PyTorch.
23. [pytorch_tutoria-quick](https://github.com/soravux/pytorch_tutorial): Quick PyTorch introduction and tutorial. Targets computer vision, graphics and machine learning researchers eager to try a new framework.  
24. [Pytorch_fine_tuning_Tutorial](https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial): A short tutorial on performing fine tuning or transfer learning in PyTorch.
25. [pytorch_exercises](https://github.com/Kyubyong/pytorch_exercises): pytorch-exercises 
26. [traffic-sign-detection](https://github.com/soumith/traffic-sign-detection-homework): nyu-cv-fall-2017 example
27. [mss_pytorch](https://github.com/Js-Mim/mss_pytorch): Singing Voice Separation via Recurrent Inference and Skip-Filtering Connections - PyTorch Implementation. Demo: js-mim.github.io/mss_pytorch
28. [DeepNLP-models-Pytorch](https://github.com/DSKSD/DeepNLP-models-Pytorch) Pytorch implementations of various Deep NLP models in cs-224n(Stanford Univ: NLP with Deep Learning)
29. [Mila introductory tutorials](https://github.com/mila-udem/welcome_tutorials): Various tutorials given for welcoming new students at MILA.
30. [pytorch.rl.learning](https://github.com/moskomule/pytorch.rl.learning): for learning reinforcement learning using PyTorch.
31. [minimal-seq2seq](https://github.com/keon/seq2seq): Minimal Seq2Seq model with Attention for Neural Machine Translation in PyTorch
32. [tensorly-notebooks](https://github.com/JeanKossaifi/tensorly-notebooks): Tensor methods in Python with TensorLy tensorly.github.io/dev
33. [pytorch_bits](https://github.com/jpeg729/pytorch_bits): time-series prediction related examples.
34. [skip-thoughts](https://github.com/sanyam5/skip-thoughts): An implementation of Skip-Thought Vectors in PyTorch.
35. [video-caption-pytorch](https://github.com/xiadingZ/video-caption-pytorch): pytorch code for video captioning. 
36. [Capsule-Network-Tutorial](https://github.com/higgsfield/Capsule-Network-Tutorial): Pytorch easy-to-follow Capsule Network tutorial.
37. [code-of-learn-deep-learning-with-pytorch](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch): This is code of book "Learn Deep Learning with PyTorch" item.jd.com/17915495606.html
38. [RL-Adventure](https://github.com/higgsfield/RL-Adventure): Pytorch easy-to-follow step-by-step Deep Q Learning tutorial with clean readable code.
39. [accelerated_dl_pytorch](https://github.com/hpcgarage/accelerated_dl_pytorch): Accelerated Deep Learning with PyTorch at Jupyter Day Atlanta II. 
40. [RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2): PyTorch4 tutorial of: actor critic / proximal policy optimization / acer / ddpg / twin dueling ddpg / soft actor critic / generative adversarial imitation learning / hindsight experience replay
41. [Generative Adversarial Networks (GANs) in 50 lines of code (PyTorch)](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f)
42. [adversarial-autoencoders-with-pytorch](https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/)
43. [transfer learning using pytorch](https://medium.com/@vishnuvig/transfer-learning-using-pytorch-4c3475f4495)
44. [how-to-implement-a-yolo-object-detector-in-pytorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
45. [pytorch-for-recommenders-101](http://blog.fastforwardlabs.com/2018/04/10/pytorch-for-recommenders-101.html)
46. [pytorch-for-numpy-users](https://github.com/wkentaro/pytorch-for-numpy-users)
47. [PyTorch Tutorial](http://www.pytorchtutorial.com/): PyTorch Tutorials in Chinese.
48. [grokking-pytorch](https://github.com/Kaixhin/grokking-pytorch): The Hitchiker's Guide to PyTorch
49. [PyTorch-Deep-Learning-Minicourse](https://github.com/Atcold/PyTorch-Deep-Learning-Minicourse): Minicourse in Deep Learning with PyTorch.
50. [pytorch-custom-dataset-examples](https://github.com/utkuozbulak/pytorch-custom-dataset-examples): Some custom dataset examples for PyTorch
51. [Multiplicative LSTM for sequence-based Recommenders](https://florianwilhelm.info/2018/08/multiplicative_LSTM_for_sequence_based_recos/)
52. [deeplearning.ai-pytorch](https://github.com/furkanu/deeplearning.ai-pytorch): PyTorch Implementations of Coursera's Deep Learning(deeplearning.ai) Specialization. 
53. [MNIST_Pytorch_python_and_capi](https://github.com/tobiascz/MNIST_Pytorch_python_and_capi): This is an example of how to train a MNIST network in Python and run it in c++ with pytorch 1.0
54. [torch_light](https://github.com/ne7ermore/torch_light): Tutorials and examples include Reinforcement Training, NLP, CV
55. [portrain-gan](https://github.com/dribnet/portrain-gan): torch code to decode (and almost encode) latents from art-DCGAN's Portrait GAN.
56. [mri-analysis-pytorch](https://github.com/omarsar/mri-analysis-pytorch): MRI analysis using PyTorch and MedicalTorch
57. [cifar10-fast](https://github.com/davidcpage/cifar10-fast): 
Demonstration of training a small ResNet on CIFAR10 to 94% test accuracy in 79 seconds as described in this [blog series](https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/).
58. [Intro to Deep Learning with PyTorch](https://in.udacity.com/course/deep-learning-pytorch--ud188): A free course by Udacity and facebook, with a good intro to PyTorch, and an interview with Soumith Chintala, one of the original authors of PyTorch.
59. [pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis): Tutorials on getting started with PyTorch and TorchText for sentiment analysis.
60. [pytorch-image-models](https://github.com/rwightman/pytorch-image-models): PyTorch image models, scripts, pretrained weights -- (SE)ResNet/ResNeXT, DPN, EfficientNet, MobileNet-V3/V2/V1, MNASNet, Single-Path NAS, FBNet, and more.
61. [CIFAR-ZOO](https://github.com/BIGBALLON/CIFAR-ZOO): Pytorch implementation for multiple CNN architectures and improve methods with state-of-the-art results. 
62. [d2l-pytorch](https://github.com/dsgiitr/d2l-pytorch): This is an attempt to modify Dive into Deep Learning, Berkeley STAT 157 (Spring 2019) textbook's code into PyTorch.
63. [thinking-in-tensors-writing-in-pytorch](https://github.com/stared/thinking-in-tensors-writing-in-pytorch): Thinking in tensors, writing in PyTorch (a hands-on deep learning intro).
64. [NER-BERT-pytorch](https://github.com/lemonhu/NER-BERT-pytorch): PyTorch solution of named entity recognition task Using Google AI's pre-trained BERT model.
65. [pytorch-sync-batchnorm-example](https://github.com/dougsouza/pytorch-sync-batchnorm-example): How to use Cross Replica / Synchronized Batchnorm in Pytorch. 
66. [SentimentAnalysis](https://github.com/barissayil/SentimentAnalysis): Sentiment analysis neural network trained by fine tuning BERT on the Stanford Sentiment Treebank, thanks to [Hugging Face](https://huggingface.co/transformers/)'s Transformers library.
67. [pytorch-cpp](https://github.com/prabhuomkar/pytorch-cpp): C++ implementations of PyTorch tutorials for deep learning researchers (based on the Python tutorials from [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)). 
68. [Deep Learning with PyTorch: Zero to GANs](https://jovian.ml/aakashns/collections/deep-learning-with-pytorch): Interactive and coding-focused tutorial series on introduction to Deep Learning with PyTorch ([video](https://www.youtube.com/watch?v=GIsg-ZUy0MY)).
69. [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch): Deep Learning with PyTorch teaches you how to implement deep learning algorithms with Python and PyTorch, the book includes a case study: building an algorithm capable of detecting malignant lung tumors using CT scans.
70. [Serverless Machine Learning in Action with PyTorch and AWS](https://www.manning.com/books/serverless-machine-learning-in-action): Serverless Machine Learning in Action is a guide to bringing your experimental PyTorch machine learning code to production using serverless capabilities from major cloud providers like AWS, Azure, or GCP.
71. [LabML NN](https://github.com/lab-ml/nn): A collection of PyTorch implementations of neural networks architectures and algorithms with side-by-side notes.
72. [Run your PyTorch Example Fedarated with Flower](https://github.com/adap/flower/tree/main/examples/pytorch_from_centralized_to_federated): This example demonstrates how an already existing centralized PyTorch machine learning project can be federated with Flower. A Cifar-10 dataset is used together with a convolutional neural network (CNN).

## Paper implementations

1. [google_evolution](https://github.com/neuralix/google_evolution): This implements one of result networks from Large-scale evolution of image classifiers by Esteban Real, et. al. 
2. [pyscatwave](https://github.com/edouardoyallon/pyscatwave): Fast Scattering Transform with CuPy/PyTorch,read the paper [here](https://arxiv.org/abs/1703.08961)
3. [scalingscattering](https://github.com/edouardoyallon/scalingscattering): Scaling The Scattering Transform : Deep Hybrid Networks.  
4. [deep-auto-punctuation](https://github.com/episodeyang/deep-auto-punctuation): a pytorch implementation of auto-punctuation learned character by character.  
5. [Realtime_Multi-Person_Pose_Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation): This is a pytorch version of Realtime_Multi-Person_Pose_Estimation, origin code is [here](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) .
6. [PyTorch-value-iteration-networks](https://github.com/onlytailei/PyTorch-value-iteration-networks): PyTorch implementation of the Value Iteration Networks (NIPS '16) paper  
7. [pytorch_Highway](https://github.com/analvikingur/pytorch_Highway): Highway network implemented in pytorch.
8. [pytorch_NEG_loss](https://github.com/analvikingur/pytorch_NEG_loss): NEG loss implemented in pytorch.  
9. [pytorch_RVAE](https://github.com/analvikingur/pytorch_RVAE): Recurrent Variational Autoencoder that generates sequential data implemented in pytorch.   
10. [pytorch_TDNN](https://github.com/analvikingur/pytorch_TDNN): Time Delayed NN implemented in pytorch.  
11. [eve.pytorch](https://github.com/moskomule/eve.pytorch): An implementation of Eve Optimizer, proposed in Imploving Stochastic Gradient Descent with Feedback, Koushik and Hayashi, 2016.  
12. [e2e-model-learning](https://github.com/locuslab/e2e-model-learning): Task-based end-to-end model learning.  
13. [pix2pix-pytorch](https://github.com/mrzhu-cool/pix2pix-pytorch): PyTorch implementation of "Image-to-Image Translation Using Conditional Adversarial Networks".   
14. [Single Shot MultiBox Detector](https://github.com/amdegroot/ssd.pytorch): A PyTorch Implementation of Single Shot MultiBox Detector.  
15. [DiscoGAN](https://github.com/carpedm20/DiscoGAN-pytorch): PyTorch implementation of "Learning to Discover Cross-Domain Relations with Generative Adversarial Networks"  
16. [official DiscoGAN implementation](https://github.com/SKTBrain/DiscoGAN): Official implementation of "Learning to Discover Cross-Domain Relations with Generative Adversarial Networks".  
17. [pytorch-es](https://github.com/atgambardella/pytorch-es): This is a PyTorch implementation of [Evolution Strategies](https://arxiv.org/abs/1703.03864) .  
18. [piwise](https://github.com/bodokaiser/piwise): Pixel-wise segmentation on VOC2012 dataset using pytorch.  
19. [pytorch-dqn](https://github.com/transedward/pytorch-dqn): Deep Q-Learning Network in pytorch.  
20. [neuraltalk2-pytorch](https://github.com/ruotianluo/neuraltalk2.pytorch): image captioning model in pytorch(finetunable cnn in branch with_finetune)
21. [vnet.pytorch](https://github.com/mattmacy/vnet.pytorch): A Pytorch implementation for V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.    
22. [pytorch-fcn](https://github.com/wkentaro/pytorch-fcn): PyTorch implementation of Fully Convolutional Networks.  
23. [WideResNets](https://github.com/xternalz/WideResNet-pytorch): WideResNets for CIFAR10/100 implemented in PyTorch. This implementation requires less GPU memory than what is required by the official Torch implementation: https://github.com/szagoruyko/wide-residual-networks .
24. [pytorch_highway_networks](https://github.com/c0nn3r/pytorch_highway_networks): Highway networks implemented in PyTorch.  
25. [pytorch-NeuCom](https://github.com/ypxie/pytorch-NeuCom): Pytorch implementation of DeepMind's differentiable neural computer paper.  
26. [captionGen](https://github.com/eladhoffer/captionGen): Generate captions for an image using PyTorch.  
27. [AnimeGAN](https://github.com/jayleicn/animeGAN): A simple PyTorch Implementation of Generative Adversarial Networks, focusing on anime face drawing. 
28. [Cnn-text classification](https://github.com/Shawn1993/cnn-text-classification-pytorch): This is the implementation of Kim's Convolutional Neural Networks for Sentence Classification paper in PyTorch.  
29. [deepspeech2](https://github.com/SeanNaren/deepspeech.pytorch): Implementation of DeepSpeech2 using Baidu Warp-CTC. Creates a network based on the DeepSpeech2 architecture, trained with the CTC activation function.
30. [seq2seq](https://github.com/MaximumEntropy/Seq2Seq-PyTorch): This repository contains implementations of Sequence to Sequence (Seq2Seq) models in PyTorch  
31. [Asynchronous Advantage Actor-Critic in PyTorch](https://github.com/rarilurelo/pytorch_a3c): This is PyTorch implementation of A3C as described in Asynchronous Methods for Deep Reinforcement Learning. Since PyTorch has a easy method to control shared memory within multiprocess, we can easily implement asynchronous method like A3C.    
32. [densenet](https://github.com/bamos/densenet.pytorch): This is a PyTorch implementation of the DenseNet-BC architecture as described in the paper Densely Connected Convolutional Networks by G. Huang, Z. Liu, K. Weinberger, and L. van der Maaten. This implementation gets a CIFAR-10+ error rate of 4.77 with a 100-layer DenseNet-BC with a growth rate of 12. Their official implementation and links to many other third-party implementations are available in the liuzhuang13/DenseNet repo on GitHub.  
33. [nninit](https://github.com/alykhantejani/nninit): Weight initialization schemes for PyTorch nn.Modules. This is a port of the popular nninit for Torch7 by @kaixhin.  
34. [faster rcnn](https://github.com/longcw/faster_rcnn_pytorch): This is a PyTorch implementation of Faster RCNN. This project is mainly based on py-faster-rcnn and TFFRCNN.For details about R-CNN please refer to the paper Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. 
35. [doomnet](https://github.com/akolishchak/doom-net-pytorch): PyTorch's version of Doom-net implementing some RL models in ViZDoom environment.  
36. [flownet](https://github.com/ClementPinard/FlowNetPytorch): Pytorch implementation of FlowNet by Dosovitskiy et al.  
37. [sqeezenet](https://github.com/gsp-27/pytorch_Squeezenet): Implementation of Squeezenet in pytorch, #### pretrained models on CIFAR10 data to come Plan to train the model on cifar 10 and add block connections too.  
38. [WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN): wassersteinGAN in pytorch. 
39. [optnet](https://github.com/locuslab/optnet): This repository is by Brandon Amos and J. Zico Kolter and contains the PyTorch source code to reproduce the experiments in our paper OptNet: Differentiable Optimization as a Layer in Neural Networks.  
40. [qp solver](https://github.com/locuslab/qpth): A fast and differentiable QP solver for PyTorch. Crafted by Brandon Amos and J. Zico Kolter.  
41. [Continuous Deep Q-Learning with Model-based Acceleration ](https://github.com/ikostrikov/pytorch-naf): Reimplementation of Continuous Deep Q-Learning with Model-based Acceleration.  
42. [Learning to learn by gradient descent by gradient descent](https://github.com/ikostrikov/pytorch-meta-optimizer): PyTorch implementation of Learning to learn by gradient descent by gradient descent.
43. [fast-neural-style](https://github.com/darkstar112358/fast-neural-style): pytorch implementation of fast-neural-style, The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with Instance Normalization.
44. [PytorchNeuralStyleTransfer](https://github.com/leongatys/PytorchNeuralStyleTransfer): Implementation of Neural Style Transfer in Pytorch. 
45. [Fast Neural Style for Image Style Transform by Pytorch](https://github.com/bengxy/FastNeuralStyle): Fast Neural Style for Image Style Transform by Pytorch .
46. [neural style transfer](https://github.com/alexis-jacq/Pytorch-Tutorials): An introduction to PyTorch through the Neural-Style algorithm (https://arxiv.org/abs/1508.06576) developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.   
47. [VIN_PyTorch_Visdom](https://github.com/zuoxingdong/VIN_PyTorch_Visdom): PyTorch implementation of Value Iteration Networks (VIN): Clean, Simple and Modular. Visualization in Visdom.  
48. [YOLO2](https://github.com/longcw/yolo2-pytorch): YOLOv2 in PyTorch.   
49. [attention-transfer](https://github.com/szagoruyko/attention-transfer): Attention transfer in pytorch, read the paper [here](https://arxiv.org/abs/1612.03928).  
50. [SVHNClassifier](https://github.com/potterhsu/SVHNClassifier-PyTorch): A PyTorch implementation of [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://arxiv.org/pdf/1312.6082.pdf).  
51. [pytorch-deform-conv](https://github.com/oeway/pytorch-deform-conv): PyTorch implementation of Deformable Convolution.  
52. [BEGAN-pytorch](https://github.com/carpedm20/BEGAN-pytorch): PyTorch implementation of [BEGAN](https://arxiv.org/abs/1703.10717): Boundary Equilibrium Generative Adversarial Networks.  
53. [treelstm.pytorch](https://github.com/dasguptar/treelstm.pytorch): Tree LSTM implementation in PyTorch.
54. [AGE](https://github.com/DmitryUlyanov/AGE): Code for paper "Adversarial Generator-Encoder Networks" by Dmitry Ulyanov, Andrea Vedaldi and Victor Lempitsky which can be found [here](http://sites.skoltech.ru/app/data/uploads/sites/25/2017/04/AGE.pdf) 
55. [ResNeXt.pytorch](https://github.com/prlz77/ResNeXt.pytorch): Reproduces ResNet-V3 (Aggregated Residual Transformations for Deep Neural Networks) with pytorch.
56. [pytorch-rl](https://github.com/jingweiz/pytorch-rl): Deep Reinforcement Learning with pytorch & visdom  
57. [Deep-Leafsnap](https://github.com/sujithv28/Deep-Leafsnap): LeafSnap replicated using deep neural networks to test accuracy compared to traditional computer vision methods.  
58. [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix): PyTorch implementation for both unpaired and paired image-to-image translation.
59. [A3C-PyTorch](https://github.com/onlytailei/A3C-PyTorch):PyTorch implementation of Advantage async actor-critic Algorithms (A3C) in PyTorch
60. [pytorch-value-iteration-networks](https://github.com/kentsommer/pytorch-value-iteration-networks): Pytorch implementation of Value Iteration Networks (NIPS 2016 best paper)  
61. [PyTorch-Style-Transfer](https://github.com/zhanghang1989/PyTorch-Style-Transfer): PyTorch Implementation of Multi-style Generative Network for Real-time Transfer
62. [pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet): pytorch-deeplab-resnet-model.
63. [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch): pytorch implementation for "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" https://arxiv.org/abs/1612.00593  
64. **[pytorch-playground](https://github.com/aaron-xichen/pytorch-playground): Base pretrained models and datasets in pytorch (MNIST, SVHN, CIFAR10, CIFAR100, STL10, AlexNet, VGG16, VGG19, ResNet, Inception, SqueezeNet)**.
65. [pytorch-dnc](https://github.com/jingweiz/pytorch-dnc): Neural Turing Machine (NTM) & Differentiable Neural Computer (DNC) with pytorch & visdom. 
66. [pytorch_image_classifier](https://github.com/jinfagang/pytorch_image_classifier): Minimal But Practical Image Classifier Pipline Using Pytorch, Finetune on ResNet18, Got 99% Accuracy on Own Small Datasets.  
67. [mnist-svhn-transfer](https://github.com/yunjey/mnist-svhn-transfer): PyTorch Implementation of CycleGAN and SGAN for Domain Transfer (Minimal).
68. [pytorch-yolo2](https://github.com/marvis/pytorch-yolo2): pytorch-yolo2
69. [dni](https://github.com/andrewliao11/dni.pytorch): Implement Decoupled Neural Interfaces using Synthetic Gradients in Pytorch
70. [wgan-gp](https://github.com/caogang/wgan-gp): A pytorch implementation of Paper "Improved Training of Wasserstein GANs".
71. [pytorch-seq2seq-intent-parsing](https://github.com/spro/pytorch-seq2seq-intent-parsing): Intent parsing and slot filling in PyTorch with seq2seq + attention
72. [pyTorch_NCE](https://github.com/demelin/pyTorch_NCE): An implementation of the Noise Contrastive Estimation algorithm for pyTorch. Working, yet not very efficient.
73. [molencoder](https://github.com/cxhernandez/molencoder): Molecular AutoEncoder in PyTorch
74. [GAN-weight-norm](https://github.com/stormraiser/GAN-weight-norm): Code for "On the Effects of Batch and Weight Normalization in Generative Adversarial Networks"
75. [lgamma](https://github.com/rachtsingh/lgamma): Implementations of polygamma, lgamma, and beta functions for PyTorch
76. [bigBatch](https://github.com/eladhoffer/bigBatch): Code used to generate the results appearing in "Train longer, generalize better: closing the generalization gap in large batch training of neural networks" 
77. [rl_a3c_pytorch](https://github.com/dgriff777/rl_a3c_pytorch): Reinforcement learning with implementation of A3C LSTM for Atari 2600. 
78. [pytorch-retraining](https://github.com/ahirner/pytorch-retraining): Transfer Learning Shootout for PyTorch's model zoo (torchvision)
79. [nmp_qc](https://github.com/priba/nmp_qc): Neural Message Passing for Computer Vision
80. [grad-cam](https://github.com/jacobgil/pytorch-grad-cam): Pytorch implementation of Grad-CAM
81. [pytorch-trpo](https://github.com/mjacar/pytorch-trpo): PyTorch Implementation of Trust Region Policy Optimization (TRPO)
82. [pytorch-explain-black-box](https://github.com/jacobgil/pytorch-explain-black-box): PyTorch implementation of Interpretable Explanations of Black Boxes by Meaningful Perturbation
83. [vae_vpflows](https://github.com/jmtomczak/vae_vpflows): Code in PyTorch for the convex combination linear IAF and the Householder Flow, J.M. Tomczak & M. Welling https://jmtomczak.github.io/deebmed.html 
84. [relational-networks](https://github.com/kimhc6028/relational-networks): Pytorch implementation of "A simple neural network module for relational reasoning" (Relational Networks) https://arxiv.org/pdf/1706.01427.pdf
85. [vqa.pytorch](https://github.com/Cadene/vqa.pytorch): Visual Question Answering in Pytorch
86. [end-to-end-negotiator](https://github.com/facebookresearch/end-to-end-negotiator): Deal or No Deal? End-to-End Learning for Negotiation Dialogues
87. [odin-pytorch](https://github.com/ShiyuLiang/odin-pytorch): Principled Detection of Out-of-Distribution Examples in Neural Networks. 
88. [FreezeOut](https://github.com/ajbrock/FreezeOut): Accelerate Neural Net Training by Progressively Freezing Layers. 
89. [ARAE](https://github.com/jakezhaojb/ARAE): Code for the paper "Adversarially Regularized Autoencoders for Generating Discrete Structures" by Zhao, Kim, Zhang, Rush and LeCun.
90. [forward-thinking-pytorch](https://github.com/kimhc6028/forward-thinking-pytorch): Pytorch implementation of "Forward Thinking: Building and Training Neural Networks One Layer at a Time" https://arxiv.org/pdf/1706.02480.pdf  
91. [context_encoder_pytorch](https://github.com/BoyuanJiang/context_encoder_pytorch): PyTorch Implement of Context Encoders
92. [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch): A PyTorch implementation of the Transformer model in "Attention is All You Need".https://github.com/thnkim/OpenFacePytorch
93. [OpenFacePytorch](https://github.com/thnkim/OpenFacePytorch): PyTorch module to use OpenFace's nn4.small2.v1.t7 model 
94. [neural-combinatorial-rl-pytorch](https://github.com/pemami4911/neural-combinatorial-rl-pytorch):  PyTorch implementation of Neural Combinatorial Optimization with Reinforcement Learning.
95. [pytorch-nec](https://github.com/mjacar/pytorch-nec): PyTorch Implementation of Neural Episodic Control (NEC)
96. [seq2seq.pytorch](https://github.com/eladhoffer/seq2seq.pytorch): Sequence-to-Sequence learning using PyTorch
97. [Pytorch-Sketch-RNN](https://github.com/alexis-jacq/Pytorch-Sketch-RNN): a pytorch implementation of arxiv.org/abs/1704.03477
98. [pytorch-pruning](https://github.com/jacobgil/pytorch-pruning): PyTorch Implementation of [1611.06440] Pruning Convolutional Neural Networks for Resource Efficient Inference
99. [DrQA](https://github.com/hitvoice/DrQA): A pytorch implementation of Reading Wikipedia to Answer Open-Domain Questions.
100. [YellowFin_Pytorch](https://github.com/JianGoForIt/YellowFin_Pytorch): auto-tuning momentum SGD optimizer
101. [samplernn-pytorch](https://github.com/deepsound-project/samplernn-pytorch): PyTorch implementation of SampleRNN: An Unconditional End-to-End Neural Audio Generation Model. 
102. [AEGeAN](https://github.com/tymokvo/AEGeAN): Deeper DCGAN with AE stabilization
103. [/pytorch-SRResNet](https://github.com/twtygqyy/pytorch-SRResNet): pytorch implementation for Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network arXiv:1609.04802v2 
104. [vsepp](https://github.com/fartashf/vsepp): Code for the paper "VSE++: Improved Visual Semantic Embeddings"
105. [Pytorch-DPPO](https://github.com/alexis-jacq/Pytorch-DPPO): Pytorch implementation of Distributed Proximal Policy Optimization: arxiv.org/abs/1707.02286
106. [UNIT](https://github.com/mingyuliutw/UNIT): PyTorch Implementation of our Coupled VAE-GAN algorithm for Unsupervised Image-to-Image Translation
107. [efficient_densenet_pytorch](https://github.com/gpleiss/efficient_densenet_pytorch): A memory-efficient implementation of DenseNets
108. [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch): Temporal Segment Networks (TSN) in PyTorch.
109. [SMASH](https://github.com/ajbrock/SMASH): An experimental technique for efficiently exploring neural architectures.
110. [pytorch-retinanet](https://github.com/kuangliu/pytorch-retinanet): RetinaNet in PyTorch
111. [biogans](https://github.com/aosokin/biogans):  Implementation supporting the ICCV 2017 paper "GANs for Biological Image Synthesis". 
112. [Semantic Image Synthesis via Adversarial Learning]( https://github.com/woozzu/dong_iccv_2017): A PyTorch implementation of the paper "Semantic Image Synthesis via Adversarial Learning" in ICCV 2017. 
113. [fmpytorch](https://github.com/jmhessel/fmpytorch): A PyTorch implementation of a Factorization Machine module in cython.
114. [ORN](https://github.com/ZhouYanzhao/ORN): A PyTorch implementation of the paper "Oriented Response Networks" in CVPR 2017. 
115. [pytorch-maml](https://github.com/katerakelly/pytorch-maml): PyTorch implementation of MAML: arxiv.org/abs/1703.03400
116. [pytorch-generative-model-collections](https://github.com/znxlwm/pytorch-generative-model-collections):  Collection of generative models in Pytorch version.
117. [vqa-winner-cvprw-2017](https://github.com/markdtw/vqa-winner-cvprw-2017): Pytorch Implementation of winner from VQA Chllange Workshop in CVPR'17. 
118. [tacotron_pytorch](https://github.com/r9y9/tacotron_pytorch):  PyTorch implementation of Tacotron speech synthesis model. 
119. [pspnet-pytorch](https://github.com/Lextal/pspnet-pytorch): PyTorch implementation of PSPNet segmentation network
120. [LM-LSTM-CRF](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF): Empower Sequence Labeling with Task-Aware Language Model http://arxiv.org/abs/1709.04109
121. [face-alignment](https://github.com/1adrianb/face-alignment): Pytorch implementation of the paper "How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)", ICCV 2017
122. [DepthNet](https://github.com/ClementPinard/DepthNet): PyTorch DepthNet Training on Still Box dataset. 
123. [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch): PyTorch version of the paper 'Enhanced Deep Residual Networks for Single Image Super-Resolution' (CVPRW 2017)
124. [e2c-pytorch](https://github.com/ethanluoyc/e2c-pytorch): Embed to Control implementation in PyTorch.
125. [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch): 3D ResNets for Action Recognition.
126. [bandit-nmt](https://github.com/khanhptnk/bandit-nmt): This is code repo for our EMNLP 2017 paper "Reinforcement Learning for Bandit Neural Machine Translation with Simulated Human Feedback", which implements the A2C algorithm on top of a neural encoder-decoder model and benchmarks the combination under simulated noisy rewards.
127. [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr): PyTorch implementation of Advantage Actor Critic (A2C), Proximal Policy Optimization (PPO) and Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation (ACKTR).
128. [zalando-pytorch](https://github.com/baldassarreFe/zalando-pytorch): Various experiments on the [Fashion-MNIST](zalandoresearch/fashion-mnist) dataset from Zalando.
129. [sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch): A PyTorch Implementation of SphereFace.
130. [Categorical DQN](https://github.com/floringogianu/categorical-dqn): A PyTorch Implementation of Categorical DQN from [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887).
131. [pytorch-ntm](https://github.com/loudinthecloud/pytorch-ntm): pytorch ntm implementation. 
132. [mask_rcnn_pytorch](https://github.com/felixgwu/mask_rcnn_pytorch): Mask RCNN in PyTorch.
133. [graph_convnets_pytorch](https://github.com/xbresson/graph_convnets_pytorch): PyTorch implementation of graph ConvNets, NIPS’16
134. [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn): A pytorch implementation of faster RCNN detection framework based on Xinlei Chen's tf-faster-rcnn.
135. [torchMoji](https://github.com/huggingface/torchMoji): A pyTorch implementation of the DeepMoji model: state-of-the-art deep learning model for analyzing sentiment, emotion, sarcasm etc.
136. [semantic-segmentation-pytorch](https://github.com/hangzhaomit/semantic-segmentation-pytorch): Pytorch implementation for Semantic Segmentation/Scene Parsing on [MIT ADE20K dataset](http://sceneparsing.csail.mit.edu)
137. [pytorch-qrnn](https://github.com/salesforce/pytorch-qrnn): PyTorch implementation of the Quasi-Recurrent Neural Network - up to 16 times faster than NVIDIA's cuDNN LSTM
138. [pytorch-sgns](https://github.com/theeluwin/pytorch-sgns): Skipgram Negative Sampling in PyTorch.
139. [SfmLearner-Pytorch ](https://github.com/ClementPinard/SfmLearner-Pytorch): Pytorch version of SfmLearner from Tinghui Zhou et al.
140. [deformable-convolution-pytorch](https://github.com/1zb/deformable-convolution-pytorch): PyTorch implementation of Deformable Convolution. 
141. [skip-gram-pytorch](https://github.com/fanglanting/skip-gram-pytorch): A complete pytorch implementation of skipgram model (with subsampling and negative sampling). The embedding result is tested with Spearman's rank correlation.
142. [stackGAN-v2](https://github.com/hanzhanggit/StackGAN-v2): Pytorch implementation for reproducing StackGAN_v2 results in the paper StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks by Han Zhang*, Tao Xu*, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas.
143. [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch): Unofficial pytorch implementation for Self-critical Sequence Training for Image Captioning. 
144. [pygcn](https://github.com/tkipf/pygcn): Graph Convolutional Networks in PyTorch.
145. [dnc](https://github.com/ixaxaar/pytorch-dnc): Differentiable Neural Computers, for Pytorch
146. [prog_gans_pytorch_inference](https://github.com/ptrblck/prog_gans_pytorch_inference): PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot.
147. [pytorch-capsule](https://github.com/timomernick/pytorch-capsule): Pytorch implementation of Hinton's Dynamic Routing Between Capsules.
148. [PyramidNet-PyTorch](https://github.com/dyhan0920/PyramidNet-PyTorch): A PyTorch implementation for PyramidNets (Deep Pyramidal Residual Networks, arxiv.org/abs/1610.02915)
149. [radio-transformer-networks](https://github.com/gram-ai/radio-transformer-networks): A PyTorch implementation of Radio Transformer Networks from the paper "An Introduction to Deep Learning for the Physical Layer". arxiv.org/abs/1702.00832
150. [honk](https://github.com/castorini/honk): PyTorch reimplementation of Google's TensorFlow CNNs for keyword spotting.
151. [DeepCORAL](https://github.com/SSARCandy/DeepCORAL): A PyTorch implementation of 'Deep CORAL: Correlation Alignment for Deep Domain Adaptation.', ECCV 2016
152. [pytorch-pose](https://github.com/bearpaw/pytorch-pose): A PyTorch toolkit for 2D Human Pose Estimation.
153. [lang-emerge-parlai](https://github.com/karandesai-96/lang-emerge-parlai): Implementation of EMNLP 2017 Paper "Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog" using PyTorch and ParlAI
154. [Rainbow](https://github.com/Kaixhin/Rainbow): Rainbow: Combining Improvements in Deep Reinforcement Learning 
155. [pytorch_compact_bilinear_pooling v1](https://github.com/gdlg/pytorch_compact_bilinear_pooling): This repository has a pure Python implementation of Compact Bilinear Pooling and Count Sketch for PyTorch.
156. [CompactBilinearPooling-Pytorch v2](https://github.com/DeepInsight-PCALab/CompactBilinearPooling-Pytorch): (Yang Gao, et al.) A Pytorch Implementation for Compact Bilinear Pooling.
157. [FewShotLearning](https://github.com/gitabcworld/FewShotLearning): Pytorch implementation of the paper "Optimization as a Model for Few-Shot Learning"
158. [meProp](https://github.com/jklj077/meProp): Codes for "meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting".
159. [SFD_pytorch](https://github.com/clcarwin/SFD_pytorch): A PyTorch Implementation of Single Shot Scale-invariant Face Detector.
160. [GradientEpisodicMemory](https://github.com/facebookresearch/GradientEpisodicMemory): Continuum Learning with GEM: Gradient Episodic Memory. https://arxiv.org/abs/1706.08840
161. [DeblurGAN](https://github.com/KupynOrest/DeblurGAN): Pytorch implementation of the paper DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks.
162. [StarGAN](https://github.com/yunjey/StarGAN): StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Tranlsation.
163. [CapsNet-pytorch](https://github.com/adambielski/CapsNet-pytorch): PyTorch implementation of NIPS 2017 paper Dynamic Routing Between Capsules.
164. [CondenseNet](https://github.com/ShichenLiu/CondenseNet): CondenseNet: An Efficient DenseNet using Learned Group Convolutions.
165. [deep-image-prior](https://github.com/DmitryUlyanov/deep-image-prior): Image restoration with neural networks but without learning.
166. [deep-head-pose](https://github.com/natanielruiz/deep-head-pose): Deep Learning Head Pose Estimation using PyTorch.
167. [Random-Erasing](https://github.com/zhunzhong07/Random-Erasing): This code has the source code for the paper "Random Erasing Data Augmentation".
168. [FaderNetworks](https://github.com/facebookresearch/FaderNetworks): Fader Networks: Manipulating Images by Sliding Attributes - NIPS 2017
169. [FlowNet 2.0](https://github.com/NVIDIA/flownet2-pytorch): FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks
170. [pix2pixHD](https://github.com/NVIDIA/pix2pixHD): Synthesizing and manipulating 2048x1024 images with conditional GANs tcwang0509.github.io/pix2pixHD 
171. [pytorch-smoothgrad](https://github.com/pkdn/pytorch-smoothgrad): SmoothGrad implementation in PyTorch
172. [RetinaNet](https://github.com/c0nn3r/RetinaNet): An implementation of RetinaNet in PyTorch.
173. [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch): This project is a faster faster R-CNN implementation, aimed to accelerating the training of faster R-CNN object detection models. 
174. [mixup_pytorch](https://github.com/leehomyc/mixup_pytorch): A PyTorch implementation of the paper Mixup: Beyond Empirical Risk Minimization in PyTorch.
175. [inplace_abn](https://github.com/mapillary/inplace_abn): In-Place Activated BatchNorm for Memory-Optimized Training of DNNs
176. [pytorch-pose-hg-3d](https://github.com/xingyizhou/pytorch-pose-hg-3d): PyTorch implementation for 3D human pose estimation
177. [nmn-pytorch](https://github.com/HarshTrivedi/nmn-pytorch): Neural Module Network for VQA in Pytorch.
178. [bytenet](https://github.com/kefirski/bytenet): Pytorch implementation of bytenet from "Neural Machine Translation in Linear Time" paper
179. [bottom-up-attention-vqa](https://github.com/hengyuan-hu/bottom-up-attention-vqa): vqa, bottom-up-attention, pytorch
180. [yolo2-pytorch](https://github.com/ruiminshen/yolo2-pytorch): The YOLOv2 is one of the most popular one-stage object detector. This project adopts PyTorch as the developing framework to increase productivity, and utilize ONNX to convert models into Caffe 2 to benifit engineering deployment.
181. [reseg-pytorch](https://github.com/Wizaron/reseg-pytorch): PyTorch Implementation of ReSeg (arxiv.org/pdf/1511.07053.pdf)
182. [binary-stochastic-neurons](https://github.com/Wizaron/binary-stochastic-neurons): Binary Stochastic Neurons in PyTorch.
183. [pytorch-pose-estimation](https://github.com/DavexPro/pytorch-pose-estimation): PyTorch Implementation of Realtime Multi-Person Pose Estimation project.
184. [interaction_network_pytorch](https://github.com/higgsfield/interaction_network_pytorch): Pytorch Implementation of Interaction Networks for Learning about Objects, Relations and Physics.
185. [NoisyNaturalGradient](https://github.com/wlwkgus/NoisyNaturalGradient): Pytorch Implementation of paper "Noisy Natural Gradient as Variational Inference". 
186. [ewc.pytorch](https://github.com/moskomule/ewc.pytorch): An implementation of Elastic Weight Consolidation (EWC), proposed in James Kirkpatrick et al. Overcoming catastrophic forgetting in neural networks 2016(10.1073/pnas.1611835114).
187. [pytorch-zssr](https://github.com/jacobgil/pytorch-zssr): PyTorch implementation of 1712.06087 "Zero-Shot" Super-Resolution using Deep Internal Learning
188. [deep_image_prior](https://github.com/atiyo/deep_image_prior): An implementation of image reconstruction methods from Deep Image Prior (Ulyanov et al., 2017) in PyTorch.
189. [pytorch-transformer](https://github.com/leviswind/pytorch-transformer): pytorch implementation of Attention is all you need.
190. [DeepRL-Grounding](https://github.com/devendrachaplot/DeepRL-Grounding): This is a PyTorch implementation of the AAAI-18 paper Gated-Attention Architectures for Task-Oriented Language Grounding
191. [deep-forecast-pytorch](https://github.com/Wizaron/deep-forecast-pytorch): Wind Speed Prediction using LSTMs in PyTorch (arxiv.org/pdf/1707.08110.pdf)
192. [cat-net](https://github.com/utiasSTARS/cat-net):  Canonical Appearance Transformations
193. [minimal_glo](https://github.com/tneumann/minimal_glo): Minimal PyTorch implementation of Generative Latent Optimization from the paper "Optimizing the Latent Space of Generative Networks"
194. [LearningToCompare-Pytorch](https://github.com/dragen1860/LearningToCompare-Pytorch): Pytorch Implementation for Paper: Learning to Compare: Relation Network for Few-Shot Learning. 
195. [poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings): PyTorch implementation of the NIPS-17 paper "Poincaré Embeddings for Learning Hierarchical Representations". 
196. [pytorch-trpo(Hessian-vector product version)](https://github.com/ikostrikov/pytorch-trpo): This is a PyTorch implementation of "Trust Region Policy Optimization (TRPO)" with exact Hessian-vector product instead of finite differences approximation.
197. [ggnn.pytorch](https://github.com/JamesChuanggg/ggnn.pytorch): A PyTorch Implementation of Gated Graph Sequence Neural Networks (GGNN). 
198. [visual-interaction-networks-pytorch](https://github.com/Mrgemy95/visual-interaction-networks-pytorch): This's an implementation of deepmind Visual Interaction Networks paper using pytorch
199. [adversarial-patch](https://github.com/jhayes14/adversarial-patch): PyTorch implementation of adversarial patch. 
200. [Prototypical-Networks-for-Few-shot-Learning-PyTorch](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch): Implementation of Prototypical Networks for Few Shot Learning (arxiv.org/abs/1703.05175) in Pytorch
201. [Visual-Feature-Attribution-Using-Wasserstein-GANs-Pytorch](https://github.com/orobix/Visual-Feature-Attribution-Using-Wasserstein-GANs-Pytorch): Implementation of Visual Feature Attribution using Wasserstein GANs (arxiv.org/abs/1711.08998) in PyTorch.
202. [PhotographicImageSynthesiswithCascadedRefinementNetworks-Pytorch](https://github.com/Blade6570/PhotographicImageSynthesiswithCascadedRefinementNetworks-Pytorch): Photographic Image Synthesis with Cascaded Refinement Networks - Pytorch Implementation
203. [ENAS-pytorch](https://github.com/carpedm20/ENAS-pytorch): PyTorch implementation of "Efficient Neural Architecture Search via Parameters Sharing". 
204. [Neural-IMage-Assessment](https://github.com/kentsyx/Neural-IMage-Assessment): A PyTorch Implementation of Neural IMage Assessment. 
205. [proxprop](https://github.com/tfrerix/proxprop): Proximal Backpropagation - a neural network training algorithm that takes implicit instead of explicit gradient steps.
206. [FastPhotoStyle](https://github.com/NVIDIA/FastPhotoStyle): A Closed-form Solution to Photorealistic Image Stylization
207. [Deep-Image-Analogy-PyTorch](https://github.com/Ben-Louis/Deep-Image-Analogy-PyTorch): A python implementation of Deep-Image-Analogy based on pytorch.
208. [Person-reID_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch): PyTorch for Person re-ID. 
209. [pt-dilate-rnn](https://github.com/zalandoresearch/pt-dilate-rnn): Dilated RNNs in pytorch. 
210. [pytorch-i-revnet](https://github.com/jhjacobsen/pytorch-i-revnet): Pytorch implementation of i-RevNets.
211. [OrthNet](https://github.com/Orcuslc/OrthNet): TensorFlow and PyTorch layers for generating Orthogonal Polynomials.
212. [DRRN-pytorch](https://github.com/jt827859032/DRRN-pytorch): An implementation of Deep Recursive Residual Network for Super Resolution (DRRN), CVPR 2017
213. [shampoo.pytorch](https://github.com/moskomule/shampoo.pytorch): An implementation of shampoo.
214. [Neural-IMage-Assessment 2](https://github.com/truskovskiyk/nima.pytorch): A PyTorch Implementation of Neural IMage Assessment.
215. [TCN](https://github.com/locuslab/TCN): Sequence modeling benchmarks and temporal convolutional networks locuslab/TCN
216. [DCC](https://github.com/shahsohil/DCC): This repository contains the source code and data for reproducing results of Deep Continuous Clustering paper.
217. [packnet](https://github.com/arunmallya/packnet): Code for PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning arxiv.org/abs/1711.05769
218. [PyTorch-progressive_growing_of_gans](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans): PyTorch implementation of Progressive Growing of GANs for Improved Quality, Stability, and Variation.
219. [nonauto-nmt](https://github.com/salesforce/nonauto-nmt): PyTorch Implementation of "Non-Autoregressive Neural Machine Translation"
220. [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN): PyTorch implementations of Generative Adversarial Networks.
221. [PyTorchWavelets](https://github.com/tomrunia/PyTorchWavelets): PyTorch implementation of the wavelet analysis found in Torrence and Compo (1998)
222. [pytorch-made](https://github.com/karpathy/pytorch-made): MADE (Masked Autoencoder Density Estimation) implementation in PyTorch
223. [VRNN](https://github.com/emited/VariationalRecurrentNeuralNetwork): Pytorch implementation of the Variational RNN (VRNN), from A Recurrent Latent Variable Model for Sequential Data.
224. [flow](https://github.com/emited/flow): Pytorch implementation of ICLR 2018 paper Deep Learning for Physical Processes: Integrating Prior Scientific Knowledge.
225. [deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch): PyTorch implementation of convolutional networks-based text-to-speech synthesis models
226. [psmm](https://github.com/elanmart/psmm): imlementation of the the Pointer Sentinel Mixture Model, as described in the paper by Stephen Merity et al.
227. [tacotron2](https://github.com/NVIDIA/tacotron2): Tacotron 2 - PyTorch implementation with faster-than-realtime inference.
228. [AccSGD](https://github.com/rahulkidambi/AccSGD): Implements pytorch code for the Accelerated SGD algorithm.
229. [QANet-pytorch](https://github.com/hengruo/QANet-pytorch): an implementation of QANet with PyTorch (EM/F1 = 70.5/77.2 after 20 epoches for about 20 hours on one 1080Ti card.)
230. [ConvE](https://github.com/TimDettmers/ConvE): Convolutional 2D Knowledge Graph Embeddings
231. [Structured-Self-Attention](https://github.com/kaushalshetty/Structured-Self-Attention): Implementation for the paper A Structured Self-Attentive Sentence Embedding, which is published in ICLR 2017: arxiv.org/abs/1703.03130 .
232. [graphsage-simple](https://github.com/williamleif/graphsage-simple): Simple reference implementation of GraphSAGE.
233. [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch): A pytorch implementation of Detectron. Both training from scratch and inferring directly from pretrained Detectron weights are available.
234. [R2Plus1D-PyTorch](https://github.com/irhumshafkat/R2Plus1D-PyTorch): PyTorch implementation of the R2Plus1D convolution based ResNet architecture described in the paper "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
235. [StackNN](https://github.com/viking-sudo-rm/StackNN): A PyTorch implementation of differentiable stacks for use in neural networks.
236. [translagent](https://github.com/facebookresearch/translagent): Code for Emergent Translation in Multi-Agent Communication.
237. [ban-vqa](https://github.com/jnhwkim/ban-vqa): Bilinear attention networks for visual question answering. 
238. [pytorch-openai-transformer-lm](https://github.com/huggingface/pytorch-openai-transformer-lm): This is a PyTorch implementation of the TensorFlow code provided with OpenAI's paper "Improving Language Understanding by Generative Pre-Training" by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
239. [T2F](https://github.com/akanimax/T2F): Text-to-Face generation using Deep Learning. This project combines two of the recent architectures StackGAN and ProGAN for synthesizing faces from textual descriptions.
240. [pytorch - fid](https://github.com/mseitzer/pytorch-fid): A Port of Fréchet Inception Distance (FID score) to PyTorch
241. [vae_vpflows](https://github.com/jmtomczak/vae_vpflows):Code in PyTorch for the convex combination linear IAF and the Householder Flow, J.M. Tomczak & M. Welling jmtomczak.github.io/deebmed.html
242. [CoordConv-pytorch](https://github.com/mkocabas/CoordConv-pytorch): Pytorch implementation of CoordConv introduced in 'An intriguing failing of convolutional neural networks and the CoordConv solution' paper. (arxiv.org/pdf/1807.03247.pdf)
243. [SDPoint](https://github.com/xternalz/SDPoint): Implementation of "Stochastic Downsampling for Cost-Adjustable Inference and Improved Regularization in Convolutional Networks", published in CVPR 2018. 
244. [SRDenseNet-pytorch](https://github.com/wxywhu/SRDenseNet-pytorch): SRDenseNet-pytorch（ICCV_2017）
245. [GAN_stability](https://github.com/LMescheder/GAN_stability): Code for paper "Which Training Methods for GANs do actually Converge? (ICML 2018)"
246. [Mask-RCNN](https://github.com/wannabeOG/Mask-RCNN): A PyTorch implementation of the architecture of Mask RCNN, serves as an introduction to working with PyTorch
247. [pytorch-coviar](https://github.com/chaoyuaw/pytorch-coviar): Compressed Video Action Recognition
248. [PNASNet.pytorch](https://github.com/chenxi116/PNASNet.pytorch): PyTorch implementation of PNASNet-5 on ImageNet. 
249. [NALU-pytorch](https://github.com/kevinzakka/NALU-pytorch): Basic pytorch implementation of NAC/NALU from Neural Arithmetic Logic Units arxiv.org/pdf/1808.00508.pdf
250. [LOLA_DiCE](https://github.com/alexis-jacq/LOLA_DiCE): Pytorch implementation of LOLA (arxiv.org/abs/1709.04326) using DiCE (arxiv.org/abs/1802.05098)
251. [generative-query-network-pytorch](https://github.com/wohlert/generative-query-network-pytorch): Generative Query Network (GQN) in PyTorch as described in "Neural Scene Representation and Rendering"
252. [pytorch_hmax](https://github.com/wmvanvliet/pytorch_hmax): Implementation of the HMAX model of vision in PyTorch.
253. [FCN-pytorch-easiest](https://github.com/yunlongdong/FCN-pytorch-easiest): trying to be the most easiest and just get-to-use pytorch implementation of FCN (Fully Convolotional Networks)
254. [transducer](https://github.com/awni/transducer): A Fast Sequence Transducer Implementation with PyTorch Bindings.
255. [AVO-pytorch](https://github.com/artix41/AVO-pytorch): Implementation of Adversarial Variational Optimization in PyTorch.
256. [HCN-pytorch](https://github.com/huguyuehuhu/HCN-pytorch): A pytorch reimplementation of { Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation }.
257. [binary-wide-resnet](https://github.com/szagoruyko/binary-wide-resnet): PyTorch implementation of Wide Residual Networks with 1-bit weights by McDonnel (ICLR 2018)
258. [piggyback](https://github.com/arunmallya/piggyback): Code for Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights arxiv.org/abs/1801.06519
259. [vid2vid](https://github.com/NVIDIA/vid2vid): Pytorch implementation of our method for high-resolution (e.g. 2048x1024) photorealistic video-to-video translation.
260. [poisson-convolution-sum](https://github.com/cranmer/poisson-convolution-sum): Implements an infinite sum of poisson-weighted convolutions
261. [tbd-nets](https://github.com/davidmascharka/tbd-nets): PyTorch implementation of "Transparency by Design: Closing the Gap Between Performance and Interpretability in Visual Reasoning" arxiv.org/abs/1803.05268 
262. [attn2d](https://github.com/elbayadm/attn2d): Pervasive Attention: 2D Convolutional Networks for Sequence-to-Sequence Prediction
263. [yolov3](https://github.com/ultralytics/yolov3): YOLOv3: Training and inference in PyTorch pjreddie.com/darknet/yolo
264. [deep-dream-in-pytorch](https://github.com/duc0/deep-dream-in-pytorch): Pytorch implementation of the DeepDream computer vision algorithm. 
265. [pytorch-flows](https://github.com/ikostrikov/pytorch-flows): PyTorch implementations of algorithms for density estimation
266. [quantile-regression-dqn-pytorch](https://github.com/ars-ashuha/quantile-regression-dqn-pytorch): Quantile Regression DQN a Minimal Working Example
267. [relational-rnn-pytorch](https://github.com/L0SG/relational-rnn-pytorch): An implementation of DeepMind's Relational Recurrent Neural Networks in PyTorch.
268. [DEXTR-PyTorch](https://github.com/scaelles/DEXTR-PyTorch): Deep Extreme Cut http://www.vision.ee.ethz.ch/~cvlsegmentation/dextr
269. [PyTorch_GBW_LM](https://github.com/rdspring1/PyTorch_GBW_LM): PyTorch Language Model for Google Billion Word Dataset.
270. [Pytorch-NCE](https://github.com/Stonesjtu/Pytorch-NCE): The Noise Contrastive Estimation for softmax output written in Pytorch
271. [generative-models](https://github.com/shayneobrien/generative-models): Annotated, understandable, and visually interpretable PyTorch implementations of: VAE, BIRVAE, NSGAN, MMGAN, WGAN, WGANGP, LSGAN, DRAGAN, BEGAN, RaGAN, InfoGAN, fGAN, FisherGAN. 
272. [convnet-aig](https://github.com/andreasveit/convnet-aig): PyTorch implementation for Convolutional Networks with Adaptive Inference Graphs.
273. [integrated-gradient-pytorch](https://github.com/TianhongDai/integrated-gradient-pytorch): This is the pytorch implementation of the paper - Axiomatic Attribution for Deep Networks.
274. [MalConv-Pytorch](https://github.com/Alexander-H-Liu/MalConv-Pytorch): Pytorch implementation of MalConv. 
275. [trellisnet](https://github.com/locuslab/trellisnet): Trellis Networks for Sequence Modeling
276. [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://github.com/minqi/learning-to-communicate-pytorch): pytorch implementation of  Learning to Communicate with Deep Multi-Agent Reinforcement Learning paper.
277. [pnn.pytorch](https://github.com/michaelklachko/pnn.pytorch): PyTorch implementation of CVPR'18 - Perturbative Neural Networks http://xujuefei.com/pnn.html.
278. [Face_Attention_Network](https://github.com/rainofmine/Face_Attention_Network): Pytorch implementation of face attention network as described in Face Attention Network: An Effective Face Detector for the Occluded Faces.
279. [waveglow](https://github.com/NVIDIA/waveglow): A Flow-based Generative Network for Speech Synthesis.
280. [deepfloat](https://github.com/facebookresearch/deepfloat): This repository contains the SystemVerilog RTL, C++, HLS (Intel FPGA OpenCL to wrap RTL code) and Python needed to reproduce the numerical results in "Rethinking floating point for deep learning" 
281. [EPSR](https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw): Pytorch implementation of [Analyzing Perception-Distortion Tradeoff using Enhanced Perceptual Super-resolution Network](https://arxiv.org/pdf/1811.00344.pdf). This work has won the first place in PIRM2018-SR competition (region 1) held as part of the ECCV 2018.
282. [ClariNet](https://github.com/ksw0306/ClariNet): A Pytorch Implementation of ClariNet arxiv.org/abs/1807.07281
283. [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT): PyTorch version of Google AI's BERT model with script to load Google's pre-trained models
284. [torch_waveglow](https://github.com/npuichigo/waveglow): A PyTorch implementation of the WaveGlow: A Flow-based Generative Network for Speech Synthesis. 
285. [3DDFA](https://github.com/cleardusk/3DDFA): The pytorch improved re-implementation of TPAMI 2017 paper: Face Alignment in Full Pose Range: A 3D Total Solution.
286. [loss-landscape](https://github.com/tomgoldstein/loss-landscape): loss-landscape Code for visualizing the loss landscape of neural nets.
287. [famos](https://github.com/zalandoresearch/famos): 
Pytorch implementation of the paper "Copy the Old or Paint Anew? An Adversarial Framework for (non-) Parametric Image Stylization" available at http://arxiv.org/abs/1811.09236.
288. [back2future.pytorch](https://github.com/anuragranj/back2future.pytorch): This is a Pytorch implementation of
Janai, J., Güney, F., Ranjan, A., Black, M. and Geiger, A., Unsupervised Learning of Multi-Frame Optical Flow with Occlusions. ECCV 2018.
289. [FFTNet](https://github.com/mozilla/FFTNet): Unofficial Implementation of FFTNet vocode paper.
290. [FaceBoxes.PyTorch](https://github.com/zisianw/FaceBoxes.PyTorch): A PyTorch Implementation of FaceBoxes.
291. [Transformer-XL](https://github.com/kimiyoung/transformer-xl): Transformer-XL: Attentive Language Models Beyond a Fixed-Length Contexthttps://github.com/kimiyoung/transformer-xl
292. [associative_compression_networks](https://github.com/jalexvig/associative_compression_networks): Associative Compression Networks for Representation Learning. 
293. [fluidnet_cxx](https://github.com/jolibrain/fluidnet_cxx): FluidNet re-written with ATen tensor lib. 
294. [Deep-Reinforcement-Learning-Algorithms-with-PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch): This repository contains PyTorch implementations of deep reinforcement learning algorithms.
295. [Shufflenet-v2-Pytorch](https://github.com/ericsun99/Shufflenet-v2-Pytorch): This is a Pytorch implementation of faceplusplus's ShuffleNet-v2. 
296. [GraphWaveletNeuralNetwork](https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork): This is a Pytorch implementation of Graph Wavelet Neural Network. ICLR 2019. 
297. [AttentionWalk](https://github.com/benedekrozemberczki/AttentionWalk): This is a Pytorch implementation of Watch Your Step: Learning Node Embeddings via Graph Attention. NIPS 2018.
298. [SGCN](https://github.com/benedekrozemberczki/SGCN): This is a Pytorch implementation of Signed Graph Convolutional Network. ICDM 2018.
299. [SINE](https://github.com/benedekrozemberczki/SINE): This is a Pytorch implementation of SINE: Scalable Incomplete Network Embedding. ICDM 2018.
300. [GAM](https://github.com/benedekrozemberczki/GAM): This is a Pytorch implementation of Graph Classification using Structural Attention. KDD 2018.
301. [neural-style-pt](https://github.com/ProGamerGov/neural-style-pt): A PyTorch implementation of Justin Johnson's Neural-style.
302. [TuckER](https://github.com/ibalazevic/TuckER): TuckER: Tensor Factorization for Knowledge Graph Completion.
303. [pytorch-prunes](https://github.com/BayesWatch/pytorch-prunes): Pruning neural networks: is it time to nip it in the bud?
304. [SimGNN](https://github.com/benedekrozemberczki/SimGNN): SimGNN: A Neural Network Approach to Fast Graph Similarity Computation.
305. [Character CNN](https://github.com/ahmedbesbes/character-based-cnn): PyTorch implementation of the Character-level Convolutional Networks for Text Classification paper. 
306. [XLM](https://github.com/facebookresearch/XLM): PyTorch original implementation of Cross-lingual Language Model Pretraining.
307. [DiffAI](https://github.com/eth-sri/diffai): A provable defense against adversarial examples and library for building compatible PyTorch models.
308. [APPNP](https://github.com/benedekrozemberczki/APPNP): Combining Neural Networks with Personalized PageRank for Classification on Graphs. ICLR 2019.
309. [NGCN](https://github.com/benedekrozemberczki/MixHop-and-N-GCN): A Higher-Order Graph Convolutional Layer. NeurIPS 2018.
310. [gpt-2-Pytorch](https://github.com/graykode/gpt-2-Pytorch): Simple Text-Generator with OpenAI gpt-2 Pytorch Implementation
311. [Splitter](https://github.com/benedekrozemberczki/Splitter): Splitter: Learning Node Representations that Capture Multiple Social Contexts. (WWW 2019).
312. [CapsGNN](https://github.com/benedekrozemberczki/CapsGNN): Capsule Graph Neural Network. (ICLR 2019).
313. [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch): The author's officially unofficial PyTorch BigGAN implementation.
314. [ppo_pytorch_cpp](https://github.com/mhubii/ppo_pytorch_cpp): This is an implementation of the proximal policy optimization algorithm for the C++ API of Pytorch.
315. [RandWireNN](https://github.com/seungwonpark/RandWireNN): Implementation of: "Exploring Randomly Wired Neural Networks for Image Recognition".
316. [Zero-shot Intent CapsNet](https://github.com/joel-huang/zeroshot-capsnet-pytorch): GPU-accelerated PyTorch implementation of "Zero-shot User Intent Detection via Capsule Neural Networks".
317. [SEAL-CI](https://github.com/benedekrozemberczki/SEAL-CI) Semi-Supervised Graph Classification: A Hierarchical Graph Perspective. (WWW 2019).
318. [MixHop](https://github.com/benedekrozemberczki/MixHop-and-N-GCN): MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing. ICML 2019.
319. [densebody_pytorch](https://github.com/Lotayou/densebody_pytorch): PyTorch implementation of CloudWalk's recent paper DenseBody.
320. [voicefilter](https://github.com/mindslab-ai/voicefilter): Unofficial PyTorch implementation of Google AI's VoiceFilter system http://swpark.me/voicefilter. 
321. [NVIDIA/semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation): A PyTorch Implementation of [Improving Semantic Segmentation via Video Propagation and Label Relaxation](https://arxiv.org/abs/1812.01593), In CVPR2019. 
322. [ClusterGCN](https://github.com/benedekrozemberczki/ClusterGCN): A PyTorch implementation of "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks" (KDD 2019).
323. [NVlabs/DG-Net](https://github.com/NVlabs/DG-Net): A PyTorch implementation of "Joint Discriminative and Generative Learning for Person Re-identification" (CVPR19 Oral). 
324. [NCRF](https://github.com/baidu-research/NCRF): Cancer metastasis detection with neural conditional random field (NCRF)
325. [pytorch-sift](https://github.com/ducha-aiki/pytorch-sift): PyTorch implementation of SIFT descriptor. 
326. [brain-segmentation-pytorch](https://github.com/mateuszbuda/brain-segmentation-pytorch): U-Net implementation in PyTorch for FLAIR abnormality segmentation in brain MRI. 
327. [glow-pytorch](https://github.com/rosinality/glow-pytorch): PyTorch implementation of Glow, Generative Flow with Invertible 1x1 Convolutions (arxiv.org/abs/1807.03039) 
328. [EfficientNets-PyTorch](https://github.com/zsef123/EfficientNets-PyTorch): A PyTorch implementation of EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
329. [STEAL](https://github.com/nv-tlabs/STEAL): STEAL - Learning Semantic Boundaries from Noisy Annotations nv-tlabs.github.io/STEAL
330. [EigenDamage-Pytorch](https://github.com/alecwangcq/EigenDamage-Pytorch): Official implementation of the ICML'19 paper "EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis".
331. [Aspect-level-sentiment](https://github.com/ruidan/Aspect-level-sentiment): Code and dataset for ACL2018 paper "Exploiting Document Knowledge for Aspect-level Sentiment Classification"
332. [breast_cancer_classifier](https://github.com/nyukat/breast_cancer_classifier): Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening arxiv.org/abs/1903.08297
333. [DGC-Net](https://github.com/AaltoVision/DGC-Net): A PyTorch implementation of "DGC-Net: Dense Geometric Correspondence Network".
334. [universal-triggers](https://github.com/Eric-Wallace/universal-triggers): Universal Adversarial Triggers for Attacking and Analyzing NLP (EMNLP 2019)
335. [Deep-Reinforcement-Learning-Algorithms-with-PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch): PyTorch implementations of deep reinforcement learning algorithms and environments.
336. [simple-effective-text-matching-pytorch](https://github.com/alibaba-edu/simple-effective-text-matching-pytorch): A pytorch implementation of the ACL2019 paper "Simple and Effective Text Matching with Richer Alignment Features".
336. [Adaptive-segmentation-mask-attack (ASMA)](https://github.com/utkuozbulak/adaptive-segmentation-mask-attack): A pytorch implementation of the MICCAI2019 paper "Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation".
337. [NVIDIA/unsupervised-video-interpolation](https://github.com/NVIDIA/unsupervised-video-interpolation): A PyTorch Implementation of [Unsupervised Video Interpolation Using Cycle Consistency](https://arxiv.org/abs/1906.05928), In ICCV 2019. 
338. [Seg-Uncertainty](https://github.com/layumi/Seg-Uncertainty): Unsupervised Scene Adaptation with Memory Regularization in vivo, In IJCAI 2020.
339. [pulse](https://github.com/adamian98/pulse): Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models
340. [distance-encoding](https://github.com/snap-stanford/distance-encoding): Distance-Encoding - Design Provably More PowerfulGNNs for Structural Representation Learning.
341. [Pathfinder Discovery Networks](https://github.com/benedekrozemberczki/PDN): Pathfinder Discovery Networks for Neural Message Passing.
342. [PyKEEN](https://github.com/pykeen/pykeen): A Python library for learning and evaluating knowledge graph embeddings.
343. [SSSNET](https://github.com/SherylHYX/SSSNET_Signed_Clustering): Official implementation of the SDM2022 paper "SSSNET: Semi-Supervised Signed Network Clustering".
344. [MagNet](https://github.com/matthew-hirn/magnet): Official implementation of the NeurIPS2021 paper "MagNet: A Neural Network for Directed Graphs".
345. [Semantic Search](https://github.com/kuutsav/information-retrieval): Latest in the field of neural information retrieval / semantic search.

## Talks & conferences

1. [PyTorch Conference 2018](https://developers.facebook.com/videos/2018/pytorch-developer-conference/): First PyTorch developer conference at 2018.

## Pytorch elsewhere

1. **[the-incredible-pytorch](https://github.com/ritchieng/the-incredible-pytorch)**: The Incredible PyTorch: a curated list of tutorials, papers, projects, communities and more relating to PyTorch. 
2. [generative models](https://github.com/wiseodd/generative-models): Collection of generative models, e.g. GAN, VAE in Tensorflow, Keras, and Pytorch. http://wiseodd.github.io  
3. [pytorch vs tensorflow](https://www.reddit.com/r/MachineLearning/comments/5w3q74/d_so_pytorch_vs_tensorflow_whats_the_verdict_on/): an informative thread on reddit. 
4. [Pytorch discussion forum](https://discuss.pytorch.org/)  
5. [pytorch notebook: docker-stack](https://hub.docker.com/r/escong/pytorch-notebook/): A project similar to [Jupyter Notebook Scientific Python Stack](https://github.com/jupyter/docker-stacks/tree/master/scipy-notebook)
6. [drawlikebobross](https://github.com/kendricktan/drawlikebobross): Draw like Bob Ross using the power of Neural Networks (With PyTorch)!
7. [pytorch-tvmisc](https://github.com/t-vi/pytorch-tvmisc): Totally Versatile Miscellanea for Pytorch
8. [pytorch-a3c-mujoco](https://github.com/andrewliao11/pytorch-a3c-mujoco): Implement A3C for Mujoco gym envs.
9. [PyTorch in 5 Minutes](https://www.youtube.com/watch?v=nbJ-2G2GXL0&list=WL&index=9).
10. [pytorch_chatbot](https://github.com/jinfagang/pytorch_chatbot): A Marvelous ChatBot implemented using PyTorch.
11. [malmo-challenge](https://github.com/Kaixhin/malmo-challenge): Malmo Collaborative AI Challenge - Team Pig Catcher
12. [sketchnet](https://github.com/jtoy/sketchnet): A model that takes an image and generates Processing source code to regenerate that image
13. [Deep-Learning-Boot-Camp](https://github.com/QuantScientist/Deep-Learning-Boot-Camp): A nonprofit community run, 5-day Deep Learning Bootcamp http://deep-ml.com. 
14. [Amazon_Forest_Computer_Vision](https://github.com/mratsim/Amazon_Forest_Computer_Vision): Satellite Image tagging code using PyTorch / Keras with lots of PyTorch tricks. kaggle competition.
15. [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku): An implementation of the AlphaZero algorithm for Gomoku (also called Gobang or Five in a Row)
16. [pytorch-cv](https://github.com/youansheng/pytorch-cv): Repo for Object Detection, Segmentation & Pose Estimation.
17. [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid): Pytorch implementation of deep person re-identification approaches.
18. [pytorch-template](https://github.com/victoresque/pytorch-template): PyTorch template project
19. [Deep Learning With Pytorch TextBook](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-pytorch) A practical guide to build neural network models in text and vision using PyTorch. [Purchase on Amazon ](https://www.amazon.in/Deep-Learning-PyTorch-practical-approach/dp/1788624335/ref=tmm_pap_swatch_0?_encoding=UTF8&qid=1523853954&sr=8-1)     [github code repo](https://github.com/svishnu88/DLwithPyTorch) 
20. [compare-tensorflow-pytorch](https://github.com/jalola/compare-tensorflow-pytorch): Compare outputs between layers written in Tensorflow and layers written in Pytorch.
21. [hasktorch](https://github.com/hasktorch/hasktorch): Tensors and neural networks in Haskell
22. [Deep Learning With Pytorch](https://www.manning.com/books/deep-learning-with-pytorch) Deep Learning with PyTorch teaches you how to implement deep learning algorithms with Python and PyTorch. 
23. [nimtorch](https://github.com/fragcolor-xyz/nimtorch): PyTorch - Python + Nim
24. [derplearning](https://github.com/John-Ellis/derplearning): Self Driving RC Car Code. 
25. [pytorch-saltnet](https://github.com/tugstugi/pytorch-saltnet): Kaggle | 9th place single model solution for TGS Salt Identification Challenge.
26. [pytorch-scripts](https://github.com/peterjc123/pytorch-scripts): A few Windows specific scripts for PyTorch.
27. [pytorch_misc](https://github.com/ptrblck/pytorch_misc): Code snippets created for the PyTorch discussion board.
28. [awesome-pytorch-scholarship](https://github.com/arnas/awesome-pytorch-scholarship): A list of awesome PyTorch scholarship articles, guides, blogs, courses and other resources.
29. [MentisOculi](https://github.com/mmirman/MentisOculi): A raytracer written in PyTorch (raynet?)
30. [DoodleMaster](https://github.com/karanchahal/DoodleMaster): "Don't code your UI, Draw it !"
31. [ocaml-torch](https://github.com/LaurentMazare/ocaml-torch): OCaml bindings for PyTorch.
32. [extension-script](https://github.com/pytorch/extension-script): Example repository for custom C++/CUDA operators for TorchScript.
33. [pytorch-inference](https://github.com/zccyman/pytorch-inference): PyTorch 1.0 inference in C++ on Windows10 platforms. 
34. [pytorch-cpp-inference](https://github.com/Wizaron/pytorch-cpp-inference): Serving PyTorch 1.0 Models as a Web Server in C++.
35. [tch-rs](https://github.com/LaurentMazare/tch-rs): Rust bindings for PyTorch.
36. [TorchSharp](https://github.com/interesaaat/TorchSharp): .NET bindings for the Pytorch engine
37. [ML Workspace](https://github.com/ml-tooling/ml-workspace): All-in-one web IDE for machine learning and data science. Combines Jupyter, VS Code, PyTorch, and many other tools/libraries into one Docker image.
38. [PyTorch Style Guide](https://github.com/IgorSusmelj/pytorch-styleguide) Style guide for PyTorch code. Consistent and good code style helps collaboration and prevents errors!


##### Feedback: If you have any ideas or you want any other content to be added to this list, feel free to contribute.
