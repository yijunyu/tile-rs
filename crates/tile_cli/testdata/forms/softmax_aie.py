from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer

def softmax(n: int = 1024):
    fifo = ObjectFifo(n, name="in")
    return Program(Runtime(), SequentialPlacer())
