import decoder
import encoder
import pywt 

def main():
	#  =      "++00100+10+-+-001-"
	st = list("CCAATAACTACGCGAATG")
	sym = {"0":"A", "1":"T", "+":"C", "-":"G"} 
	
	d = StackRunDecoder(sym)
	d.decode(st)


if __name__ == "__main__":
	main()