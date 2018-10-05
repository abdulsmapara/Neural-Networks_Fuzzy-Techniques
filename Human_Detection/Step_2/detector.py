import cv2
import numpy as np
from random import seed
from random import random
from math import exp
import os

from PIL import Image

#from matplotlib import pyplot as plt


def split_images(path):
    img = cv2.imread(path)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 450, 290)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    # plt.imshow(img),plt.colorbar(),plt.show()
    cv2.imwrite(path, img)


def detect_edges(path):  # currently assumes one object in frame but once segregated will work fine

    img = cv2.imread(path, 0)
    edges = cv2.Canny(img, 30, 40)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=7)

    # cv2.imshow('window',dilation)
    # print path
    cv2.imwrite(path, dilation)


# cv2.imshow('window2',edges)


# cv2.waitKey(0)

def generate_feature_vector(path):  # currently based on shape only

    image = cv2.imread(path)
    # resize to 9*9
    resized_image = cv2.resize(image, (15, 15))
    cv2.imwrite(path, resized_image)

    im = Image.open(path).convert('RGB')
    pix = im.load()
    w = im.size[0]
    h = im.size[1]
    count = 0
    ret = []
    print("genearting feature vector...")
    for i in range(w):
        for j in range(h):
            rval = pix[i, j][0]
            gval = pix[i, j][1]
            bval = pix[i, j][2]
            val = 0
            if rval == 0 and gval == 0 and bval == 0:
                #print("0")
                val=0
            elif rval == 255 and gval == 255 and bval == 255:
                #print("1")
                val = 1
            else:
                # diff1 = 255 - rval
                # diff2 = 255 - gval
                # diff3 = 255 - bval
                if rval < 10 and gval < 10 and bval < 10:
                    #print("0")
                    val=0
                elif rval > 150 and gval > 150 and bval > 150:
                   # print("1")
                    val = 1
                elif (rval > 100 and gval > 100) or (rval > 100 and bval > 100) or (gval > 100 and bval > 100):
                    #print("1")
                    val = 1
                else:
                    #print("0")
                    val=0
            ret.append(val)
            count = count + 1

    print("count is {}".format(count))

    print("")
    return ret


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

print("Scanning images folder ...")

dataset = []
total_count = 0
even_odd = 1
for filename in os.listdir("images"):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith("png"):
        print("Scanning {}".format(filename))
        split_images("images/" + filename)
        detect_edges("images/" + filename)
        res = []
        res = generate_feature_vector("images/" + filename)
        res.append(even_odd)  # expected output
        if even_odd == 1:
            even_odd = 0
        else:
            even_odd = 1
        dataset.append(res)
        total_count = total_count + 1
'''
for filename in os.listdir("Non_Human"):
   if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith("png"):
			print ("Scanning {}".format(filename))
			split_images("Non_Human/"+filename)
			detect_edges("Non_Human/"+filename)
			res=[]
			res=generate_feature_vector("Non_Human/"+filename)
			res.append(0)	#expected output
			dataset.append(res)
			total_count = total_count + 1

'''
print(dataset)

num_hidden = 5
num_inputs = len(dataset[0]) - 1
num_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(num_inputs, num_hidden, num_outputs)
final_weights = []
train_network(network, dataset, 0.5, 200, num_outputs)
print("")
for layer in network:
    for neuron in layer:
        final_weights.append(neuron['weights'])
        print(neuron['output'])
print("")
print("")
'''Fetching weights and bias from network'''
modify_network = []
for layer in network:
    temp = []
    for d in layer:
        dic = {}
        dic['weights'] = d['weights']
        temp.append(dic)
    modify_network.append(temp)


for layer in modify_network:
    print(layer)
    print("\n")

test_dataset = []
even_odd=1
for filename in os.listdir("test"):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith("png"):
        print("Scanning {}".format(filename))
        split_images("test/" + filename)
        detect_edges("test/" + filename)
        res = []
        res = generate_feature_vector("test/" + filename)
        feed = int(input("Enter prediction for {} ".format(filename)))
        res.append(feed)
        test_dataset.append(res)


#scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
#print('Scores: %s' % scores)
#print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

correct=0
total=0
print("Final Result:")

for row in test_dataset:
    prediction = predict(network, row)
    print("Expected =%d Got=%d" % (row[-1],prediction))
    if row[-1] == prediction:
        correct=correct+1
    total = total + 1

print("Accuracy: {}".format(1.0*correct/total))


'''Storing Weight matrix'''
np.save("weights_after_training", modify_network)