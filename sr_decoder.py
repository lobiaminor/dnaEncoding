import sys

class StackRunDecoder(object):
	"""A stack run decoder using a base 4 alphabet. 

	Attributes:
		symbols: dictionary containing the equivalences between our bases and the regular ones - the keys (0,1,+,-)
	"""

	def __init__(self, symbols):
		"""Return a Decoder object"""
		# Swap keys and values in the dict, we need it that way
		self.symbols = {v:k for k,v in symbols.items()}


	def decode(self, qits):
		"""Decodes an incoming list of qits 
		Params:
		bits: stack-run encoded image, represented by a list of bits
		"""
		run_sym   = ["+", "-"]
		stack_sym = ["0", "1"]

		# Convert the string to base 0,1,+,-
		qits = list(map(self.symbols.get, qits))
		
		# Determine if we will begin with a run or with a stack
		mode = "RUN" if (qits[0] in run_sym) else "STACK"  
		
		# Current word we are trying to decode
		current = []

		# Output
		result = []
		
		for q in qits:
			if mode == "RUN":
				# As long as we keep receiving + or -, store them and continue
				if q in run_sym:
					current.append(q)
				# When we receive 0 or 1, clear current, send the run to the decoder
				# and start storing the stack
				else:
					# Only decode the run if there is actually one (could be 0)
					if len(current) > 0:
						# Add as many zeros as the length of the run
						result.extend(["0"]*self.decode_run(current))
					# Clear the buffer
					current.clear()
					current.append(q)
					mode = "STACK"

			elif mode == "STACK":
				# If we receive a + or -
				if q in stack_sym:
					current.append(q)
				else:
					# Append the 0 or 1, decode the stack and clear current
					current.append(q)
					result.append(str(self.decode_stack(current)))
					current.clear()
					mode = "RUN"

		print(result)


	def decode_stack(self, bits):
		"""Decodes a bit string (ending with a sign) representing an encoded stack"""
		
		# Stacks are stored in reverse order, so first of all switch it
		bits.reverse()
		
		# Check if positive or negative TODO: the + shouldn't be hardcoded
		sign = 1 if(bits[0] == "+") else -1
		
		# The first bit should be a 1
		bits[0] = "1"
		
		# Convert back to decimal
		bits = "".join(bits)
		return sign * (int(bits, 2)-1)


	def decode_run(self, bits):
		"""Decodes a bit string (ending with a sign) representing an encoded run"""

		# If the string contains only +'s, it means it is 2^k - 1, 
		# so we don't need to add a + in the end. Otherwise we do
		if not(len(set(bits)) == 1 and bits[0] == "+"): # TODO: The + shouldn't be hardcoded here either 
			bits.append("+")

		# Runs are saved in reverse order too, so undo it
		bits.reverse()
		
		# Substitute + and -  for 1 and 0 and convert binary to int
		return int("".join(bits).replace("+", "1").replace("-", "0"), 2)


def main():
	#  =      "++00100+10+-+-001-"
	st = list("CCAATAACTACGCGAATG")
	sym = {"0":"A", "1":"T", "+":"C", "-":"G"} 
	
	d = StackRunDecoder(sym)
	d.decode(st)

if __name__ == "__main__":
	main()






