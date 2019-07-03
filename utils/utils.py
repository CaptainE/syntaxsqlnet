import torch
import numpy as np

def calculate_mask(lengths,max_len, batch_size, device=torch.device('cpu')):

	
    #Convert to tensor so we can compare
    lengths = torch.tensor(lengths,dtype=torch.long)

    #calculate mask
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    
    mask = mask.unsqueeze(2) #[batch_size, max_len, 1]

    return mask

	
def length_to_mask(lengths, lengths2=None, max_len=None, max_len2=None, device=torch.device('cpu')):
    """
    Converts list of lengths to a byte mask array

    Args:
        lengths [List] : list of lengths
        lengths2 [List]: list of lengths for second sequence
        max_len [Int] : maximum length of sequence 1
        max_len2 [Int] : maximum length of sequence 2
    Returns:
        mask [batch_size, max_len, 1] or [batch_size, max_len, max_len2] : Byte mask for input sequence(s)
    """
	
	#Find max length of lengths
    max_len = max_len or int(max(lengths))
    batch_size = len(lengths)
	
    mask = calculate_mask(lengths,max_len, batch_size)

    if lengths2 is not None:
        #Find max length of lengths
        max_len2 = max_len2 or int(max(lengths2))
		
        mask2 = calculate_mask(lengths2,max_len2, batch_size)
        mask2 = mask2.expand(batch_size, max_len2, max_len)
        mask2 = mask2.permute(0,2,1)

        mask = mask.expand(batch_size, max_len, max_len2) #[batch_size, max_len, max_len2]

        mask = mask & mask2
    	
    #masked_fill_ requires the mask to be a byte (uint8) tensor
    mask = mask.byte()

    return mask

	
def pad(sentences, pad_token=0):

    #Calculat the sequence length of the batch
    lengths = [len(sentence) for sentence in sentences]
    max_len = max(lengths)

    padded = []
    for example in sentences:
        #calculate how much padding is needed for this example
        pads = max_len - len(example)
		
        #Pad with zeros
        padded.append(np.pad(example, ((0,pads)), 'constant', constant_values=pad_token))
		
    padded = np.asarray(padded)
    lengths = np.asarray(lengths)
	
    return padded, lengths



if __name__ == '__main__':

    lengths = [1,4,6]
    lengths2 = [2,2,1]
    length_to_mask(lengths, lengths2, device=torch.device('cpu'))
    