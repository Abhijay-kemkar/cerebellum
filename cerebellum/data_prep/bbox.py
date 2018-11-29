from cerebellum.utils.mask import get_bbox

class BboxDict(dict):
	"""Bounding box of an object in a segmentation"""
	def __init__(self):
		"""
		Attributes:
			obj_id (int): object label
			bbox (6x, array): bbox co-ordinates in order Z_min, Y_min, X_min, Z_max, Y_max, X_max
		"""
		dict.__init__(self)