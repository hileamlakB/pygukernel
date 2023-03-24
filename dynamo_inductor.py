# import builtins
# import traceback

# def custom_print(*args, **kwargs):
#     call_stack = traceback.extract_stack(limit=2)  # Get the call stack
#     caller = call_stack[0]  # Get the caller's information
#     builtins.original_print(f"[DEBUG] Called from {caller.filename} at line {caller.lineno}")
    
#     # Call the original print function with the provided arguments
#     builtins.original_print(*args, **kwargs)

# # Replace the built-in print function with the custom one
# builtins.original_print = builtins.print
# builtins.print = custom_print


# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch._inductor.config as config
import torch._dynamo as dynamo
config.debug = True


def test_model(model, *args):
    # compile model using inductor
    compiled_model = torch.compile(model)
    # run model
    y_pred = compiled_model(*args)
    # optimize on prediction
    try:
        y_pred.sum().backward()
    except RuntimeError as e:
        pass

# models
relu = torch.nn.Sequential(
    torch.nn.Linear(3, 10),
    torch.nn.ReLU()
).cuda()

linear = torch.nn.Sequential(
    torch.nn.Linear(1, 1)
).cuda()
    
test_model(relu, torch.randn(100, 3).cuda())

# @dynamo.optimize('inductor')
# def multiple_relu(a, b):
#     return torch.matmul(a, b)

# x = torch.randn(100, 3).cuda()
# y = torch.randn(3, 10).cuda()
# multiple_relu(x, y)
    



   # compute / io
    # how many fuses
    # try reduction like max average, and see if they are fused
    # try to fuse with other ops, see how many would fuse
    # convultion, matrix multiply, addmm
    # point wise, see if the are fused
    # kernal size
    # 10 mn to 16 gb
    # metrix vector multipliction, metrix multpilict, pointwise, and reduction
    # we need to learn the mapping from convoluition to matrix multiplication