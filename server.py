'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''
import numpy as np
from flask import Flask, request
from flask import render_template
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.special import softmax
import time
import json
import matplotlib
matplotlib.use('TkAgg')


app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''

    sample_points_X, sample_points_Y = [], []
    # TODO: Start sampling (12 points)

    #Finding the total distance between gesture points
    tot = 0
    for j in range(1,len(points_X)):
        dist = abs(euclidean([points_X[j],points_Y[j]] , [points_X[j-1],points_Y[j-1]]))
        tot+=dist

    #Sampling points between every pair of gesture points based on the distance between them proportional to the total distance traversed by the gesture
    for j in range(1,len(points_X)):
        dist = abs(euclidean([points_X[j-1],points_Y[j-1]], [points_X[j], points_Y[j]]))
        if(dist==0):
            sample_points_X.append(np.linspace(points_X[j],points_X[j-1],num=1))
            sample_points_Y.append(np.linspace(points_Y[j],points_Y[j-1],num=1))
            continue
        offset = int(((dist/tot) * 100))
        XC = np.linspace(points_X[j-1],points_X[j],endpoint=True,num=offset)
        sample_points_X.append(XC)

        YC = np.linspace(points_Y[j-1],points_Y[j],endpoint=True,num=offset)
        sample_points_Y.append(YC)

    #Flatenning list of lists to a single list
    flattenedX = [val for sublist in sample_points_X for val in sublist]
    flattenedY = [val for sublist in sample_points_Y for val in sublist]

    #Appending the computed points to the output list
    if(len(flattenedX)<100 and len(flattenedX)>0):
        x = 100 - len(flattenedX)
        varx = flattenedX[-1]
        vary = flattenedY[-1]
        for i in range(x):
            flattenedX.append(varx+0.03)
            flattenedY.append(vary+0.02)

    #print(len(flattenedY), len(flattenedX))
    return np.asarray(flattenedX[:100]), np.asarray(flattenedY[:100])

# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    threshold = 20
    # TODO: Do pruning (12 points)

    #Computing the distances between the first and last points of the gesture and sample points
    for i in range(len(template_sample_points_X)):

        # start point distance
        if euclidean([gesture_points_X[0],gesture_points_Y[0]],[template_sample_points_X[i][0],template_sample_points_Y[i][0]] ) > threshold:
            continue
        # end point distance
        if euclidean([gesture_points_X[-1],gesture_points_Y[-1]],[template_sample_points_X[i][-1],template_sample_points_Y[i][-1]] ) > threshold:
            continue
        #Appending the valid words and valid templates to the respective output lists
        valid_template_sample_points_X.append(np.copy(template_sample_points_X[i]))
        valid_template_sample_points_Y.append(np.copy(template_sample_points_Y[i]))
        valid_words.append(words[i])

    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    shape_scores = []
    # TODO: Set your own L
    L = 3

    # TODO: Calculate shape scores (12 points)
    #Normalising the input gesture lists
    gesW = max(gesture_sample_points_X) - min(gesture_sample_points_X)
    gesH = max(gesture_sample_points_Y) - min(gesture_sample_points_Y)
    centx = gesW/2
    centy = gesH/2
    gesture_sample_points_X = [abs(i-centx) for i in gesture_sample_points_X]
    gesture_sample_points_Y = [abs(i - centy) for i in gesture_sample_points_Y]

    #Scaling the gesture lists based on the scaling factor computed
    gesW = max(gesture_sample_points_X) - min(gesture_sample_points_X)
    gesH = max(gesture_sample_points_Y) - min(gesture_sample_points_Y)
    scoreges = L/max(gesH,gesW)
    gesture_sample_points_Y =  np.multiply(gesture_sample_points_Y, scoreges)
    gesture_sample_points_X = np.multiply(scoreges, gesture_sample_points_X)

    gesture_point_pairs = np.stack((gesture_sample_points_X, gesture_sample_points_Y), axis=-1)
    s = []
    templatenewX = []
    templatenewY = []

    #Normalising and Scaling the valid template lists based on the scaling factor computed
    for i in range(0,len(valid_template_sample_points_X)):
        temW = max(valid_template_sample_points_X[i]) - min(valid_template_sample_points_X[i])
        temH = max(valid_template_sample_points_Y[i]) - min(valid_template_sample_points_Y[i])
        centxt = temW/2
        centyt = temH/2

        if(temH==0 or temW==0):
            d = 1.0
        else:
            d = L/max(temH,temW)
        templatenewX.append(np.multiply(d, valid_template_sample_points_X[i]))
        templatenewY.append(np.multiply(d,valid_template_sample_points_Y[i]))

    #Computing the shape scores for the gesture with respect to the valid template sample points
    for i in range(len(valid_template_sample_points_X)):
            template_point_pairs = np.stack((templatenewX[i], templatenewY[i]), axis=-1)
            # shape_scores.append(np.sum(scipy.spatial.distance.cdist(gesture_point_pairs, template_point_pairs, 'euclidean').diagonal())/100.0)
            shape_scores.append(np.sum(np.linalg.norm(np.subtract(gesture_point_pairs, template_point_pairs), ord=2, axis=1.))/100.0)
    return shape_scores

ALPHA = softmax(np.concatenate([np.linspace(1.0, 0.001, num=50), np.linspace(0.001, 1.0, num=50)]))
def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

        In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
        template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 15
    # TODO: Calculate location scores (12 points)
    #Computing the location scores for the gesture with respect to the valid template sample points
    gesture_point_pairs = np.stack((gesture_sample_points_X, gesture_sample_points_Y), axis=-1)
    for i in range(len(valid_template_sample_points_X)):
        template_point_pairs = np.stack((valid_template_sample_points_X[i], valid_template_sample_points_Y[i]), axis=-1)
        # Dut = np.sum([max(0, min(euclidean_distances(template_point_pairs,[ges_point]))[0] - radius) for ges_point in gesture_point_pairs])
        Dut = np.sum([max(0, min(np.linalg.norm(np.subtract(template_point_pairs, [ges_point]), ord=2, axis=1.)) - radius) for ges_point in gesture_point_pairs])
        # Dtu = np.sum([max(0, min(euclidean_distances(gesture_point_pairs,[tpl_point]))[0] - radius) for tpl_point in template_point_pairs])
        Dtu = np.sum([max(0, min(np.linalg.norm(np.subtract(gesture_point_pairs, [tpl_point]), ord=2, axis=1.)) - radius) for tpl_point in template_point_pairs])
        if Dtu == 0 and Dut == 0 :
            location_scores.append(0)
        else:
           location_scores.append(np.sum(ALPHA * np.linalg.norm(np.subtract(gesture_point_pairs, template_point_pairs), ord=2, axis=1.)))
    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.8
    # TODO: Set your own location weight
    location_coef = 0.7
    #Computing the integration scores for the gesture
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = 'the'
    # TODO: Set your own range.
    n = 5
    # TODO: Get the best word (12 points)

    #Selecting the best word based on the integration scores
    top_n_idxs = np.argsort(integration_scores)[0:n]
    integ_scores = [integration_scores[i] for i in top_n_idxs]
    top_n_words = [valid_words[i] for i in top_n_idxs]
    top_n_words_withscores = zip(top_n_words, integ_scores)
    if len(top_n_words) == 0:
        return 'No matches'
    return min(top_n_words_withscores, key=lambda x: x[1])[0]


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])


    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'

if __name__ == "__main__":
    app.run()