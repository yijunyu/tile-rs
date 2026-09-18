import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def softmax(in_tensor, out_tensor):
    row = nl.load(in_tensor)
    nl.store(out_tensor, nl.exp(row - nl.max(row)))
