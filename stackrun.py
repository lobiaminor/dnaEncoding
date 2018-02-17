class StackRunDecoder(object):
    """A stack run decoder using a base 4 alphabet. 

    Attributes:
        symbols: dictionary containing the equivalences between our bases and the regular ones - the keys (0,1,+,-)
    """

    def __init__(self, symbols):
        """Return a Decoder object"""
        # Swap keys and values in the dict, we need it that way
        self.symbols = {v:k for k,v in symbols.items()}


    def decode(self, signal):
        """Decodes an incoming list of signal 
        Params:
        - signal: stack-run encoded image, represented by a list of bits
        """
        run_sym   = ["+", "-"]
        stack_sym = ["0", "1"]

        # Convert the string to base 0,1,+,-
        signal = list(map(self.symbols.get, signal))
        
        # Determine if we will begin with a run or with a stack
        mode = "RUN" if (signal[0] in run_sym) else "STACK"  
        
        # Current word we are trying to decode
        current = []

        # Output
        result = []
        
        for q in signal:
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
                        result.extend([0]*self.decode_run(current))
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
        
        # If there is something left on current, it means a run is yet to be decoded
        if current:
            result.extend([0]*self.decode_run(current))

        return result


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
        return int(sign * (int(bits, 2)-1))


    def decode_run(self, bits):
        """Decodes a bit string (ending with a sign) representing an encoded run"""

        # If the string contains only +'s, it means it is 2^k - 1,
        # so we don't need to add a + in the end. Otherwise we do
        if not(len(set(bits)) == 1 and bits[0] == "+"):
            bits.append("+")

        # Runs are saved in reverse order too, so undo it
        bits.reverse()
        
        # Substitute + and -  for 1 and 0 and convert binary to int
        return int("".join(bits).replace("+", "1").replace("-", "0"), 2)


class StackRunEncoder(object):

    def __init__(self, symbols):
        self.symbols = symbols


    def stackEncoding(self, stack_value):
        # calculate the +/- representation for the msb
        if stack_value < 0:
            msb = "-"
        else:
            msb = "+"
        # we work with the asolute value of stack_value
        stack_value = abs(stack_value)
        # the algorithm requires the number to be encoded + 1
        stack_value += 1
        # get the bit representation of the stack value (+1) and store each bit in list element
        bit_rep = list("{0:b}".format(int(stack_value)))
        # substitute the most significant bit with its +/- representation
        bit_rep[0] = msb
        # invert the order of the obtained list
        bit_rep = bit_rep[::-1]

        return bit_rep


    def runEncoding(self, run_length):
        # we encode 0 as -, 1 as + and we truncate the least significant bit if the number is not of the form (2^k)-1
        bit_rep = list("{0:b}".format(run_length))
        for b in bit_rep:
            if b == "0":
                b = "-"
            else:
                b = "+"

        # invert the order of the obtained list
        bit_rep = bit_rep[::-1]
        # we can remove the last element of the list UNLESS the encoded value contains only "+"
        for b in bit_rep:
            if b == "-":
                del bit_rep[-1]
                break

        return bit_rep


    def encode(self, signal):
        run_length = 0
        encoded_sig = list()
        
        for s in signal:
            if s == 0:
                run_length+=1
            else:
                encoded_sig.extend(self.runEncoding(run_length))
                run_length = 0
                encoded_sig.extend(self.stackEncoding(s))

        # Handle the case of having a run at the end of the signal
        if signal[-1] == 0:
            encoded_sig.extend(self.runEncoding(run_length)) 

        encoded_translated_sig = list(map(self.symbols.get, encoded_sig))

        return encoded_translated_sig