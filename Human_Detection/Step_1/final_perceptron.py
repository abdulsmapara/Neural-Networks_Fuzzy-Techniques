from numpy import array,dot

input_size=45
output_size=45
max_size=45

def predict(row, weights):	#Calculates yin and returns y
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	if activation > 0.0:
		return 1.0
	elif activation == 0.0:
		return 0.0
	else:
		return -1.0

def train_weights(train, l_rate):
	weights = [0.0 for i in range(len(train[0]))]
	print(weights)
	epoch = 1
	brk = False
	prev_error = 0.00
	while brk==False and epoch <= 15:
		brk=True
		succ = 0
		tot=0
		brk = True
		#sum_error = 0.0
		for row in train:
			#print('row')
			#print(row)
			prediction = predict(row, weights)
			#error = row[-1] - prediction
			#sum_error += error**2		
			if prediction != row[max_size]:
				weights[0] = weights[0] + l_rate * row[max_size]
			for i in range(len(row)-1):
				if prediction != row[max_size]:
					weights[i + 1] = weights[i + 1] + l_rate * row[max_size]  * row[i]	#update weights
					#print('upcoming weights ')
					#print(weights[i+1])
					brk=False
				else:
					succ=succ+1
				tot=tot+1
		print('Accuracy ')
		print((1.0*succ)/tot)
		print('*epoch=%d, lrate=%.3f' % (epoch, l_rate))
		epoch = epoch + 1
		#print(weights)
	return weights

training_data = [[1,1,1],[-1,1,1],[1,-1,1],[-1,-1,-1]]
training_data1 = [[1,1,1],[-1,1,1],[1,-1,1],[-1,-1,-1]]

#training data for human detection
dataset = [[0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],[0,0,1,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,1],[0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0],[0,1,1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,1]]
#trial ----Just for clarity\\
vec1=[0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1]
vec2=[0,0,1,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,1]
vec3=[0,1,1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,1]
vec66=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
vec77=[0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0]
vec88=[0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0]

#test on training data itself
tes1=[1,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0]#insan.jpg
tes2=[1,0,0,1,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0]#insan2.jpg
tes3=[1,0,1,1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0]#insan3.jpg
tes77=[1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0]#car.jpg
tes88=[1,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,0,0,1,0,0,0]#car.jpeg

testing_data=[1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,0]#rohit_sharma.jpg


#final weight matrix obtained here hardcoded
final_wt=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]

tesla=[1,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0]#insan3_new.jpg (back side)
tiger=[1,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,0,0,1,0,0,0]#tiger.jpg
l_rate = 1
final = train_weights(dataset, l_rate)
print(final)
#print(dot(vec77,final))



#print(dot(vec3,final))

#print(dot(vec2,final))

#print(dot(vec1,final))

#Threshold value is 20.0

#Testing on training data
threshold = 20.0
if dot(tes1,final) >= threshold:	#human
	print('Detected')
else:
	print("Human not detected")
if dot(tes2,final) >= threshold:	#human
	print('Detected')
else:
	print("Human not detected")
if dot(tes3,final) >= threshold:	#human
	print('Detected')
else:
	print("Human not detected")

if dot(tes77,final) >= threshold:
	print('Detected')
else:
	print("Human not detected")
if dot(tes88,final) >= threshold:	#not human car
	print('Detected')
else:
	print('Human not detected')

#---------------------Testing on unseen data-------------
if dot(tesla,final) >= threshold:	#human
	print('Unseen Testing input - human detected')
else:
	print("Human not detected in unseen testing")


if dot(tiger,final) >= threshold:	#not human
	print('Unseen Testing input - human detected')
else:
	print("Human not detected in unseen testing")


if dot(testing_data,final)>=threshold:	#human
	print('Unseen image-human Detected')
else:
	print('Unseen image Human not detected')