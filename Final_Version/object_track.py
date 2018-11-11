# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
 
class CentroidTracker():
	def __init__(self, maxDisappeared=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
 
		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared
		def register(self, centroid):
			# when registering an object we use the next available object
			# ID to store the centroid
			self.objects[self.nextObjectID] = centroid
			self.disappeared[self.nextObjectID] = 0
			self.nextObjectID += 1
		def deregister(self, objectID):
			# to deregister an object ID we delete the object ID from
			# both of our respective dictionaries
			del self.objects[objectID]
			del self.disappeared[objectID]
		def update(self, rects):
			# check to see if the list of input bounding box rectangles
			# is empty
			if len(rects) == 0:
				# loop over any existing tracked objects and mark them
				# as disappeared
				for objectID in self.disappeared.keys():
					self.disappeared[objectID] += 1
	 
					# if we have reached a maximum number of consecutive
					# frames where a given object has been marked as
					# missing, deregister it
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
	 
				# return early as there are no centroids or tracking info
				# to update
				return self.objects
		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")
 
		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)