### Variational Mapping of MRI:
We introduce and demonstrate a new paradigm for quantitative parameter mapping in MRI.
Parameter mapping techniques, such as diffusion MRI (dMRI) and quantitative MRI (qMRI), have the potential to robustly and repeatably measure biologically-relevant tissue maps that strongly relate to underlying microstructure.
Quantitative maps are calculated by fitting a model to multiple images, e.g. with least-squares or machine learning. However, the overwhelming majority of model fitting techniques assume that each voxel is independent, ignoring any co-dependencies in the data. This makes model fitting sensitive to voxelwise measurement noise,  hampering reliability and repeatability.
We propose a self-supervised deep variational autoencoder  model fitting approach that breaks the assumption of independent pixels, leveraging redundancies in the data to effectively perform data-driven regularisation of quantitative maps.  
We demonstrate that our approach outperforms current model fitting techniques in dMRI simulations and real data.
Our approach enables improved quantitative maps and/or reduced acquisition times, and can hence support the clinical adoption of parameter mapping methods such as dMRI and qMRI.


[To do] Refactorizing the code for the real MRI data.

### Installation:
```shell
conda create --name myenv \

conda activate myenv \

conda install --file requirement.txt
```

### Simulated Experiments:

```shell
python train_simulated.py --config-file 'configs/baseline_simulated.yaml' 
```

### Results:

The visual results of the proposed methods (VAE-UniG and VAE-GMM) vs the non-dl and self-supervised baseline. Our methods can discover more anatomical structures as highlighted in the zoomed in areas.

<br>
 <img height="500" src="figures/visual_results_zoom.png" />
</br>