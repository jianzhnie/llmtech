## Data Parallel Training

PyTorch provides several options for data-parallel training. For applications that gradually grow from simple to complex and from prototype to production, the common development trajectory would be:

1. Use single-device training, if the data and model can fit in one GPU, and the training speed is not a concern.
2. Use single-machine multi-GPU [DataParallel](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html), if there are multiple GPUs on the server, and you would like to speed up training with the minimum code change.
3. Use single-machine multi-GPU [DistributedDataParallel](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html), if you would like to further speed up training and are willing to write a little more code to set it up.
4. Use multi-machine [DistributedDataParallel](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html) and the [launching script](https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md), if the application needs to scale across machine boundaries.
5. Use [torchelastic](https://pytorch.org/elastic) to launch distributed training, if errors (e.g., OOM) are expected or if the resources can join and leave dynamically during the training.



### 1.  torch.nn.DataParallel

The [DataParallel](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html) package enables single-machine multi-GPU parallelism with the lowest coding hurdle. It only requires a one-line change to the application code. The tutorial [Optional: Data Parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html) shows an example. The caveat is that, although `DataParallel` is very easy to use, it usually does not offer the best performance. This is because the implementation of `DataParallel` replicates the model in every forward pass, and its single-process multi-thread parallelism naturally suffers from GIL contentions. To get better performance, please consider using [DistributedDataParallel](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html).

####  1.1 torch.nn.DataParallel

```python
class torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
    """
    Parameters
    module (Module) – module to be parallelized

    device_ids (list of python:int or torch.device) – CUDA devices (default: all devices)

    output_device (int or torch.device) – device location of output (default: device_ids[0])
    """
```

Examples

```python
>>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
>>> output = net(input_var)  # input_var can be on any device, including CPU
```

Implements data parallelism at the module level.

This container parallelizes the application of the given `module` by splitting the input across the specified devices by chunking in the batch dimension (other objects will be copied once per device). In the forward pass, the module is replicated on each device, and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.

The batch size should be larger than the number of GPUs used.



Arbitrary positional and keyword inputs are allowed to be passed into DataParallel but some types are specially handled. tensors will be **scattered** on dim specified (default 0). tuple, list and dict types will be shallow copied. The other types will be shared among different threads and can be corrupted if written to in the model’s forward pass.

The parallelized `module` must have its parameters and buffers on `device_ids[0]` before running this [`DataParallel`](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) module.

#### WARNING

> In each forward, `module` is **replicated** on each device, so any updates to the running module in `forward` will be lost. For example, if `module` has a counter attribute that is incremented in each `forward`, it will always stay at the initial value because the update is done on the replicas which are destroyed after `forward`. However, [`DataParallel`](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) guarantees that the replica on `device[0]` will have its parameters and buffers sharing storage with the base parallelized `module`. So **in-place** updates to the parameters or buffers on `device[0]` will be recorded. E.g., [`BatchNorm2d`](https://pytorch.org/docs/master/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d) and [`spectral_norm()`](https://pytorch.org/docs/master/generated/torch.nn.utils.spectral_norm.html#torch.nn.utils.spectral_norm) rely on this behavior to update the buffers.



#### WARNING

>Forward and backward hooks defined on `module` and its submodules will be invoked `len(device_ids)` times, each with inputs located on a particular device. Particularly, the hooks are only guaranteed to be executed in correct order with respect to operations on corresponding devices. For example, it is not guaranteed that hooks set via [`register_forward_pre_hook()`](https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook) be executed before all `len(device_ids)` [`forward()`](https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.forward) calls, but that each such hook be executed before the corresponding [`forward()`](https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.forward) call of that device.





Data Parallelism is when we split the mini-batch of samples into multiple smaller mini-batches and run the computation for each of the smaller mini-batches in parallel.

Data Parallelism is implemented using `torch.nn.DataParallel`. One can wrap a Module in `DataParallel` and it will be parallelized over multiple GPUs in the batch dimension.
