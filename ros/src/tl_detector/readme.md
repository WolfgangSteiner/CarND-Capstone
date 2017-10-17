
The model is based purely on CNNs and trained on 64x64 image samples which the architure transforms into 1x1x4 which is exactly the feature vector red yellow green nothing. Running it on larger samples leads to bigger x and y dimensions leading to a heatmap image. The details on the training can be found in the jupyter notebook.

The node is implemented a little bit different compared to the basecode. Every time the node gets an image the traffic light coordinates are transformed to car coordinates to find which one is currently likely to be detected. Then inference is run on the image and if there are more than 5 detections above a threshold (there should always be multiple as there are many 64x64 tiles with a traffic light if there is one in the image) the one with the most detections is chosen. If that is red, the index of the waypoint is published.

Test inference implementation in the node todo: add code to find the light in the list and publish
other things
yellow lights are unrelyable because limited training samples

base images for training (source resized to 400x300)
https://drive.google.com/open?id=0B-HSlrDXC6xCa1cyalV5Z1ExTTA

Actual samples used for training (64x64)
https://drive.google.com/open?id=0B-HSlrDXC6xCU0RIZGt1cHFIV1E
