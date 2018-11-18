import time

class TimingDecorators(object):
	"""Decorators to get run time of function"""
	@classmethod
	def print_runtime(func):
		def runtime_wrapper(*args, **kwargs):
			start_time = time.time()
			result = func(*args, **kwargs)
			runtime = time.time()-start_time
			print func.__name__ + " runtime: %f s"%(runtime)
			return result
		return runtime_wrapper