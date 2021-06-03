# Product Detection of Shelfs:

In this project we aim to detect all the different products available on a shoping store shelf/ cigarette vending machine

## Execution details :

-
  ### Directory structure :
  ```
  examplefolder/Production-Detection (this project)
  examplefolder/GroceryDataset_part1 ( https://github.com/gulvarol/grocerydataset and replace GroceryDataset_part1/ShelfImages with https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz )
  ```
  ### Execution steps :
    - install requirements
    - Load the given ipynb file to jupiter note book and execute the cells
    - If the model is not trained i.e. if any of yolov2.ckpt.data-00000-of-00001, yolov2.ckpt.index, yolov2.ckpt.meta and checkpoint is missing, then uncomment #cell 22 and run the file

      OR
    - install requirements
    - run ``` python3 product_detection.py ```
    - If the model is not trained i.e. if any of yolov2.ckpt.data-00000-of-00001, yolov2.ckpt.index, yolov2.ckpt.meta and checkpoint is missing, then uncomment lines 724 to 741 and execute the file

Output for the execution is stored in image2products.json and metrics.json, where image2products.json has the {(filename):(number of products detected)...} info and metrics.json has mAP, precision and recall values.
## Dataset preparation :
- Scaling images
- bboxes

## Detection Network :

Used YOLOv2 (You only look once) architecture because Yolov2 has very good accuracy as it removes the Fully Connected Layers and instead made a Fully Convoultional Network with anchor boxes which helped detect bounding boxes better.

## Results

Calculated mAP over 8 iou thresholds [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75].
And evaluated precission and recall for iou threshold of 0.5

- mAP: 0.8403684210526317
- precision: 0.8023032629558541
- recall: 0.88

## References :

https://github.com/JongsooKeum/yolov2-tensorflow


