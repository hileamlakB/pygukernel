import time
import csv
from . import ir, dependencies as deps


class Benchmark:
    
    is_created = False
    
    def __init__(self) -> None:
        # self.nodes = nodes
        self.original_code = None
        self.triton_code = None
        self.time_stamp = time.time()
        self.result = []
        self.called = 0
        self.nodes = None
        self.read_writes = None
        
    def run_benchmark(self):
        """Run triton code and return compute time"""
        print("running benchmark")
        
    
    def writelines(self):
        
        assert(self.nodes and self.read_writes and self.original_code and self.triton_code)
        
        # create a file for original code
        oc_f = f"original_code_{self.called}_{self.time_stamp}.txt"
        with open(oc_f, 'w') as f:
            f.writelines(self.original_code)
        
        # create a file for triton code
        tc_f = f"triton_code_{self.called}_{self.time_stamp}.py"
        with open(tc_f, 'w') as f:
            f.writelines(self.triton_code)
        
        self.run_benchmark()
        
        
        for node, read_write in zip(self.nodes, self.read_writes):
            
            reads = ""
            for read in read_write.reads:
                if isinstance(read, deps.MemoryDep):
                    reads += f"{read.size};"
            writes = ""
            for write in read_write.writes:
                if isinstance(write, deps.MemoryDep):
                    writes += f"{write.size};"
            
            # print(node)
            # Write extracted properties
            if isinstance(node, ir.ComputedBuffer):
                inputs = ""
                output = f"{node.layout.dtype}, {node.layout.size}, {node.layout.stride}"
                self.result.append(["ComputedBuffer", node.name, inputs, output, node.origins, reads, writes, 0, oc_f, tc_f])
            if isinstance(node, ir.ExternKernel):
                inputs = ""
                for  in_ in node.inputs:
                    inputs +=  f"{in_.layout.dtype}, {in_.layout.size}, {in_.layout.stride};"
                output = f"{node.layout.dtype}, {node.layout.size}, {node.layout.stride}"
                self.result.append(["ExternKernel", node.name, inputs, output, node.origins, reads, writes, 0, oc_f, tc_f])

        # print(self.result)
        self.called += 1
        self.nodes = None
        self.read_writes = None
        self.original_code = None
        self.triton_code = None
                    
    def write_code(self):
        print("writing code")
    
    def dump(self):
        
        file = f"benchmark_{self.time_stamp}.csv"
        with open(file, 'w', newline='') as f:
            
            csv_writer = csv.writer(f)
            # Write header row
            csv_writer.writerow(["Node Type", "Node Name", "Inputs", "Output", "Function", "Compute_time", "Reads", "Writes", "Original Code",  "Triton Code"])
            
            for row in self.result:
                csv_writer.writerow(row)
            

            
# create a new benchmark
if not Benchmark.is_created:
    print("creating benchmark")
    Benchmark.is_created = True
    benchmark = Benchmark()
    
