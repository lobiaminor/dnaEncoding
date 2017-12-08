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
        bit_rep = list("{0:b}".format(stack_value))
        # substitute the most significant bit with its +/- representation
        bit_rep[0] = msb
        # invert the order of the obtained list
        bit_rep = bit_rep[::-1]

        return bit_rep


    def runEncoding(self, run_length):
        # we encode 0 as -, 1 as + and we truncate the least significant bit if the number is not of the form (2^k)-1
        bit_rep = list("{0:b}".format(run_length))
        for i in range(0, len(bit_rep)):
            if bit_rep[i] == "0":
                bit_rep[i] = "-"
            else:
                bit_rep[i] = "+"

        # invert the order of the obtained list
        bit_rep = bit_rep[::-1]
        # we can remove the last element of the list UNLESS the encoded value contains only "+"
        for i in range(0, len(bit_rep)):
            if bit_rep[i] == "-":
                del bit_rep[-1]
                break

        return bit_rep


    def encode(self, signal):
        run_length = 0
        encoded_sig = list()
        
        run_freqs = {}
        stack_freqs = {}
        
        for s in signal:
            if s == 0:
                run_length+=1
            else:
                codeword = self.runEncoding(run_length)
                encoded_sig.extend(codeword)
                codeword = "".join(codeword) # Convert it to string so we can use it as key for the dict
                if codeword in run_freqs:
                    run_freqs[codeword] = run_freqs[codeword] + 1
                else:
                    run_freqs[codeword] = 1 
                run_length = 0
                codeword = self.stackEncoding(s)
                encoded_sig.extend(codeword)
                codeword = "".join(codeword)
                if codeword in stack_freqs:
                    stack_freqs[codeword] = stack_freqs[codeword] + 1
                else:
                    stack_freqs[codeword] = 1 

        # Handle the case of having a run at the end of the signal
        if signal[-1] == 0:
            codeword = self.runEncoding(run_length)
            encoded_sig.extend(codeword)
            codeword = "".join(codeword)
            if codeword in run_freqs:
                run_freqs[codeword] = run_freqs[codeword] + 1
            else:
                run_freqs[codeword] = 1 

        encoded_translated_sig = list(map(self.symbols.get, encoded_sig))

        return encoded_translated_sig, run_freqs, stack_freqs