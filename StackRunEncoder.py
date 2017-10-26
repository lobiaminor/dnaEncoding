class StackRunEncoder(object):
    def __init__(self, symbols):
        self.symbols = symbols

    def StackEncoding (stack_value):

        # calculate the +/- representation for the msb
        if stack_value < 0:
            msb = "-"
        else:
            msb = "+"
        #we work with the asolute value of stack_value
        stack_value = abs(stack_value)
        #the algorithm requires the number to be encoded + 1
        stack_value+=1
        #get the bit representation of the stack value (+1) and store each bit in list element
        bit_rep = list("{0:b}".format(stack_value))
        #substitute the most significant bit with its +/- representation
        bit_rep[0] = msb
        #invert the order of the obtained list
        bit_rep = bit_rep[::-1]

        return bit_rep





    def RunEncoding (run_length):
        #we encode 0 as -, 1 as + and we truncate the least significant bit if the number is not of the form (2^k)-1
        bit_rep = list("{0:b}".format(run_length))
        for i in range(0, len(bit_rep)):
            if bit_rep[i] == "0":
                bit_rep[i] = "-"
            else:
                bit_rep[i] = "+"

        #invert the order of the obtained list
        bit_rep = bit_rep[::-1]
        #we can remove the last element of the list UNLESS the encoded value contains only "+"
        for i in range(0, len(bit_rep)):
            if bit_rep[i] == "-":
                del bit_rep[-1]
                break

        return bit_rep

    def encode(signal):
        run_length = 0
        encoded_sig = list()
        for i in range(0, len(signal)):
            if signal[i] == 0:
                run_length+=1
            else:
                encoded_sig.extend(RunEncoding(run_length))
                run_length = 0
                encoded_sig.extend(StackEncoding(signal[i]))

        print(encoded_sig)
        encoded_translated_sig = list(map(self.symbols.get, encoded_sig))
        print(encoded_translated_sig)
        return encoded_translated_sig