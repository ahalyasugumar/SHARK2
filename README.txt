README:

The following functions were executed as decribed below in order to implement the SHARK algorithm for gestures,

1. Sampling: This function samples 100 equidistant points on the gesture for processing
 

2. Pruning: Sets a threshold to limit the valid word templates that need to be used for processing. Also identifies the valid words based on the theshold defined.


3. Shape Score: The gesture and the valid templates are normalised and scaled based on their respective scaling factos

 

4. Location Score: The sampled user gesture (containing 100 points) is compared with every single valid template (containing 100 points) and each of them is given a location score.
 

5. Integration Score: Computes the integration scores for all the sample valid words so that we can output the smallest integration score as our final output.

HYPER PARAMETERS CONFIGURED:

1. threshold = 20
2. L = 3
3. radius = 15
4. ALPHA = softmax(np.concatenate([np.linspace(1.0, 0.001, num=50), np.linspace(0.001, 1.0, num=50)]))
5. shape_coef = 0.8
6. location_coef = 0.7
7. n = 5