Things to try



## Task-related

1. Run current model without classification and see what the loss is for different parameters. Use it to choose the best one.
2. Add support for categorical data.
  a. This could be a great case for semi-supervised.
  b. I should really give each set of of categorical variables embedding or MLP encoders and add them together or use their concatenated outputs as inputs to the data.
    1. Then, to semi-supervised, add a network that predicts the categorical output data.
3. Add a discriminator to tell the difference between real and false samples.


### Architecture-related

1. Add an attention mechanism
2. Add residual connetions
3. Try the ResiDual idea for transformers
4. Minor: changing location of batchnorm



### Application-related
1. Make a training class that tracks residuals. Could be used for stats, outlier detection, and importance sampling.
2. Statistical testing.
  A. Incorporate PyDESeq2 to outputs of the neural network. First, for analysis of cells, and second, for significance testing between groups (pseudobulk)



### General ideas

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9977174/

They model cell type data and layer data. They infer cell tpye and layer shift at the same time. They wsih they had a model that predicted both. Might be unstable, they will try in the future.

Interesting ideas.
Another key feature of POLARIS is its ability to leverage image data. In the ST deconvolution field, gene expression itself has proved its ability to infer cell composition. Imaging information, however, has been underused. Recent work (18, 19) showed the potential of histology images accompanying ST data. We believe that histology images can be further leveraged for ST inference. For example, histology images alone are widely used to segment cells with deep learning models (13, 14). POLARIS can take an accompanying image as input to train an image network and use a pretrained image network on a completely new image. Our pretrained POLARIS image network offers a novel method for tissue registration, which extracts and reveals tissue anatomical or functional structures either from the histological image alone or jointly with gene expression. The major barriers that prevent the full potential of integrating histological images with ST data include quality of the co-registered image and, most importantly, the absence of pathologist annotations. To accomplish the task with the currently available data, POLARIS incorporates training of the image network into the inference of cell composition. Instead of training a model with inferred cell composition as the goal and using MSE as the loss function, our intention is to use histology images to help with the estimation of cell composition under the rationale that spots with similar histological images and similar neighborhoods tend to share similar cell composition. Nevertheless, we fail to demonstrate that the image improves deconvolution performance due to the limitation of the current data: Single cell–level resolution ST data only provide DAPI-stained images, which only comprise one color panel, while spot-level ST data have no gold standard truth. Despite these limitations to quantify the performance, POLARIS with image input still achieved high accuracy among the state-of-the-art methods in single-cell resolution ST data, and in spot-level ST data, cell type composition inferred by POLARIS agreed with single-cell level data and the expected biological layers (e.g., glutamatergic neurons in the mouse cortex and cancer epithelial cells in the breast cancer slides). POLARIS introduces a novel approach for inferring cell composition purely from histological images that has not previously been explored by ST deconvolution. We believe that the versatile ability of POLARIS to incorporate histological images to elucidate layer-specific gene expression patterns will empower discoveries in spatial biology.


I think a deep learning model that tries to predict unperterbed and stuff could do good. Latent diffusion is a good idea.

https://github.com/CODAIT/deep-histopath has some good ideas - identifying where tissue is
