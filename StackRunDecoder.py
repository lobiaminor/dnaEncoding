import sys

# class StackRunDecoder(object):
#     """A stack run decoder using a base 4 alphabet. 

#     Attributes:
#         cod: dictionary containing the equivalences between our bases and the regular ones - the keys (0,1,+,-)
#     """

#     def __init__(self, name, balance=0.0):
#         """Return a Customer object whose name is *name* and starting
#         balance is *balance*."""
#         self.name = name
#         self.balance = balance


def decode(qits):
	"""Decodes an incoming list of qits 
	Params:
	bits: stack-run encoded image, represented by a list of bits
	"""
	run_sym   = ["+", "-"]
	stack_sym = ["0", "1"]

	# Determine if we will begin with a run or with a stack
	mode = "RUN" if (qits[0] in run_sym) else "STACK"  
	current = []  # Current word we are trying to decode	
	
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
					print("  RUN: " + str(decode_run(current)))
				# Clear the buffer
				current.clear()
				current.append(q)
				mode = "STACK"

		elif mode == "STACK":
			# If we receive a + or -
			if q in stack_sym:
				current.append(q)
			else:
				current.append(q)
				#decode_stack(current)
				print("STACK: " + str(decode_stack(current)))
				current.clear()
				mode = "RUN"

def decode_stack(bits):
	"""Decodes a bit string (ending with a sign) representing an encoded stack"""
	
	# Stacks are stored in reverse order, so first of all switch it
	bits.reverse()
	# Check if positive or negative TODO: the + shouldn't be hardcoded
	symbol = 1 if(bits[0] == "+") else -1
	# The first bit should be a 1
	bits[0] = "1" 
	return symbol * (int("".join(bits), 2)-1)

def decode_run(bits):
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
	s = "++00100+10+-+-001-"
	decode(s)

if __name__ == "__main__":
	main()






