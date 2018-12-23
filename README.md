# cerebellum

This repository contains tools to generate segmentation of parallel fibers from electron microscopy images of neuronal tissue. In addition to scripts to run the traditional connectomics pipeline ([affnity generation](https://github.com/donglaiw/EM-network.git), [region agglomeration](https://github.com/donglaiw/waterz.git)), it provides tools for:
* Error analysis based on voxel-wise comparison and skeleton comparison
* Segmentation stitching using IoU based object-tracking
* Basic error correction for false splits and false merges

## Getting started

Clone the repository.
```git clone --recursive https://github.com/srujanm/cerebellum.git```

Clone the [ibex](https://github.com/srujanm/ibex.git) repository which contains skeletonization tools.
```
cd cerebellum/
git clone --recursive https://github.com/srujanm/cerebellum.git
```
Follow the installation instructions on the [ibex](https://github.com/srujanm/ibex.git) page.

Depending on your requirement, install packages from our traditional EM pipeline. 
* [EM-network](https://github.com/srujanm/EM-network.git) for affinity prediction
* [waterz](https://github.com/srujanm/waterz.git) for region agglomeration
* [EM-segLib](https://github.com/donglaiw/EM-seglib.git) for error analysis

## Scripts

The `scripts` folder is organized into various sub-folders based on the stage of operation in the pipeline. Here are quick descriptions of them from start to end of the pipeline.

* `network`: training and testing of affinity generation networks
* `waterz`: region agglomeration from affinity and generation of segmentation
* `gt_prep`: preparation of ground truth segmentation blocks and skeletons
* `seg_prep`: preparation and error analysis of predicted segmentation blocks
* `seg_prep_unetfiber`: preparation and error analysis of predicted segmentation blocks from new affinity network trials
* `blockchain`: linking of segmentation blocks and assembly of full volume
* `tracking`: basic error correction
* `pnj_scripts`: scripts used to refine Purkinje cell segmentation

## Notebooks

The `notebooks` provide tutorial-style introductions to various modules in the repository.

* [Skeleton generation](https://github.com/srujanm/cerebellum/blob/master/notebooks/skeletons_helloworld.ipynb)
* [Block level error analysis](https://github.com/srujanm/cerebellum/blob/master/notebooks/block_level_analysis.ipynb)
* [Block stitching error analysis](https://github.com/srujanm/cerebellum/blob/master/notebooks/erroranalysis_block2block.ipynb)
* [Scaling of errors with tissue volume](https://github.com/srujanm/cerebellum/blob/master/notebooks/superblock_error_scaling.ipynb)
* [False merge correction](https://github.com/srujanm/cerebellum/blob/master/notebooks/merge_correction_trial.ipynb)
* [Comparison of affinity networks](https://github.com/srujanm/cerebellum/blob/master/notebooks/erl_comparison.ipynb)