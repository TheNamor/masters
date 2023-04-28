Files:
    r models.R
        This is the file I ran my final analysis in. I  did some basic data exploration to determine
        which models I thought would work the best.
        I ran SVM models, RandomForest models, and nearest neighbor models.
    Melanoma_cellgraph_globalfeats_functions.py
        This file I only slightly modified, I disabled some features that were not significant for our
        dataset.
    generate_feats.py
        This was the main file that read the centroids in, generated the cell-graphs, found the features,
        and saved the output. As the input file was so huge (>2 million rows), I used a generator to only
        read the file one sample at a time, to prevent overloading the program's memory. I then used the
        coordinates of the centroids to generate the necessary edges, and converted it into a graph. I then
        used this graph as an input to the slightly modified Melanoma_cellgraph_globalfeats_functions file.
        I then saved the features to an output csv file.
    get_labels.py
        I used this small script to generate the appropriate labels for the 2-class and 3-class classification
    stardist_trained.py
        This file went through 2 main iterations. The first iteration was used to train and optimize the Stardist
        model. This included loading in the Consep images, patchifying them, reading the labels, and then beginning
        the training. Once the model was done training it was saved. The second iteration used this saved model to
        label the CRC dataset, and label the images. The CRC images needed to be loaded in, patchified, and labeled.
        I then found the physical properties of each patch, and stitched them back together to generate the full
        dataset, which I then saved to a csv.
    stardist_pre.py
        I used this file to test out a pretrained stardist model, as a quick proof of concept and to see whether a
        fully trained model was required. The performance wasn't amazing but showed promise so I moved forward
        with training my own. I also used this file to try out different patch sizes to see if it affected performance.
    unet.py
        This file contains the Unet model I started with as a proof of concept. It achieved a meanIOU of 0.707 on
        the Consep dataset which was a good sign for using a deep learning approach for instance segmentation.
    watershed.py
        This file runs through the watershed process to further clean the instance segmentations. I used it on the
        output of the Unet model I started with. The output from Stardist was already clean enough that additional
        watershed was not required.
    ex1-3.png
        These images show predictions by the Stardist model for samples from the Consep training dataset. As you can see
        the cells are cleanly separated and closely resemble the label
    test1-4.png
        These images show predictions by the Stardist model for samples from the Consep validation dataset. Even
        in the test set the predictions are very close to the true labels.