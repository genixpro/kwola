"""CPU, CUDA, and NCCL diagnostics for fresh Kwola experiments."""

import os
import shutil
import tempfile
import traceback
import uuid
from pathlib import Path

import torch
import torch.distributed
import torch.multiprocessing

from ..components.agents.DeepLearningAgent import DeepLearningAgent
from ..config.config import KwolaCoreConfiguration


def _new_test_config():
    config_dir = KwolaCoreConfiguration.createNewLocalKwolaConfigDir(
        "testing", url="http://127.0.0.1:3003/", email="", password="", name="", paragraph="",
        enableTypeEmail=True, enableTypePassword=True, enableRandomNumberCommand=False,
        enableRandomBracketCommand=False, enableRandomMathCommand=False,
        enableRandomOtherSymbolCommand=False, enableDoubleClickCommand=False,
        enableRightClickCommand=False,
    )
    return config_dir, KwolaCoreConfiguration.loadConfigurationFromDirectory(config_dir)


def _run_step(config, gpu=None, world_size=1):
    agent = DeepLearningAgent(config=config, whichGpu=gpu, gpuWorldSize=world_size)
    agent.initialize(enableTraining=True)
    results = agent.learnFromBatches([agent.prepareEmptyBatch()], trainingStepIndex=100)
    # The first eleven tuple entries are aggregate losses; the remaining
    # fields include batch metadata and per-sample arrays.
    if not results or any(not torch.isfinite(torch.as_tensor(value)).all() for value in results[0][:11]):
        raise RuntimeError("Kwola training diagnostic produced a NaN or infinite loss")
    agent.save()


def _nccl_worker(rank, world_size, coordinator_path):
    config_dir, config = _new_test_config()
    try:
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="file://" + coordinator_path,
            world_size=world_size, rank=rank,
        )
        _run_step(config, gpu=rank, world_size=world_size)
        torch.distributed.barrier()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        shutil.rmtree(config_dir, ignore_errors=True)


def testTwoGPUNCCL(verbose=True):
    """Exercise a coordinated two-rank forward/backward/optimizer step."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Two CUDA devices are required for the NCCL diagnostic; found %d." % torch.cuda.device_count())
    capabilities = [torch.cuda.get_device_capability(index) for index in range(2)]
    if capabilities != [(7, 5), (7, 5)]:
        raise RuntimeError("Expected two RTX 2070 (compute capability 7.5) devices, found %r." % capabilities)
    coordinator = str(Path(tempfile.gettempdir()) / ("kwola-nccl-" + uuid.uuid4().hex))
    if verbose:
        print("Running a two-rank NCCL forward/backward/optimizer diagnostic on CUDA devices 0 and 1.")
    try:
        torch.multiprocessing.spawn(_nccl_worker, args=(2, coordinator), nprocs=2, join=True)
    finally:
        Path(coordinator).unlink(missing_ok=True)


def testNeuralNetworkAllGPUs(verbose=True, require_two_gpus=False):
    """Run CPU and single-GPU diagnostics; optionally require the dual-GPU gate."""
    config_dir, config = _new_test_config()
    try:
        if verbose:
            print("Running CPU forward/backward/optimizer diagnostic.")
        _run_step(config, gpu=None)
        if torch.cuda.is_available():
            if verbose:
                print("Running single-GPU forward/backward/optimizer diagnostic on CUDA device 0.")
            _run_step(config, gpu=0)
        elif require_two_gpus:
            raise RuntimeError("CUDA is unavailable; the dual-GPU diagnostic cannot run.")
    except Exception:
        if verbose:
            traceback.print_exc()
        return False
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)

    if require_two_gpus:
        try:
            testTwoGPUNCCL(verbose=verbose)
        except Exception:
            if verbose:
                traceback.print_exc()
            return False
    return True
