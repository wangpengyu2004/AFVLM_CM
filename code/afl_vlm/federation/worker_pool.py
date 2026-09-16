"""Spawned single-GPU client workers for one federated experiment.

The authoritative server and Method instance stay in the parent process.  A
worker owns one full model replica, receives an immutable dispatch package,
runs only client-side hooks, and returns a raw update.  This follows the
server/scheduler/trainer separation used by APPFL and FLGo while preserving the
project's deterministic virtual event order.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from afl_vlm.federation.types import ClientContext, ClientSpec, ScheduledEvent, Update
from afl_vlm.models.base import LoRAState, TrainConfig, nested_to_device, state_to_device


@dataclass(slots=True)
class TrainJob:
    event: ScheduledEvent
    context: ClientContext
    base_state: LoRAState
    start_state: LoRAState
    train_config: TrainConfig
    method_runtime_state: dict[str, Any]
    group_size: int | None
    dispatch_wall_time: float

    @property
    def job_id(self) -> str:
        return f"train:{self.event.event_id}"


@dataclass(slots=True)
class EvaluationJob:
    job_id: str
    states: dict[str, LoRAState]
    split: str


@dataclass(frozen=True, slots=True)
class ShutdownWorker:
    pass


@dataclass(slots=True)
class FreshStateResponse:
    request_id: str
    state: LoRAState
    version: int


@dataclass(slots=True)
class WorkerReady:
    worker_id: int
    device_id: int
    initial_state: LoRAState | None
    parameter_names: tuple[str, ...]


@dataclass(slots=True)
class WorkerStatus:
    worker_id: int
    message: str


@dataclass(slots=True)
class WorkerProgress:
    worker_id: int
    job_id: str
    client_id: str
    current: int
    total: int
    loss: float


@dataclass(slots=True)
class FreshStateRequest:
    worker_id: int
    event_id: int
    request_id: str


@dataclass(slots=True)
class TrainCompleted:
    worker_id: int
    event_id: int
    update: Update


@dataclass(slots=True)
class EvaluationCompleted:
    worker_id: int
    job_id: str
    metrics: dict[str, Any]


@dataclass(slots=True)
class WorkerFailed:
    worker_id: int
    job_id: str
    traceback_text: str


WorkerCommand = TrainJob | EvaluationJob | FreshStateResponse | ShutdownWorker
WorkerMessage = (
    WorkerReady
    | WorkerStatus
    | WorkerProgress
    | FreshStateRequest
    | TrainCompleted
    | EvaluationCompleted
    | WorkerFailed
)


def update_to_device(update: Update, device: Any) -> Update:
    update.base_state = state_to_device(update.base_state, device)
    update.local_state = state_to_device(update.local_state, device)
    update.delta = state_to_device(update.delta, device)
    if update.mean_gradient is not None:
        update.mean_gradient = state_to_device(update.mean_gradient, device)
    return update


def _evaluate_states(
    model: Any,
    data_module: Any,
    states: Mapping[str, LoRAState],
    split: str,
    status: Any,
) -> dict[str, Any]:
    tasks = data_module.tasks
    sample_ids = {
        task: [sample.id for sample in adapter.load_split(split)] for task, adapter in tasks.items()
    }
    if set(states) == {"global"}:
        model.load_trainable(states["global"])
        metrics = {}
        for task, adapter in tasks.items():
            status(f"evaluating {task} ({split})")
            model.set_evaluation_context(task, None)
            metrics[task] = model.evaluate(adapter, sample_ids[task], "final")
        return metrics
    by_task: dict[str, list[dict[str, float]]] = {}
    for client_id, state in states.items():
        task = client_id.split("/", 1)[0]
        status(f"evaluating {client_id} ({split})")
        model.load_trainable(state)
        model.set_evaluation_context(task, client_id)
        by_task.setdefault(task, []).append(
            model.evaluate(tasks[task], sample_ids[task], "final")
        )
    return {
        task: {metric: sum(row[metric] for row in rows) / len(rows) for metric in rows[0]}
        for task, rows in by_task.items()
    }


def _worker_main(
    worker_id: int,
    device_id: int,
    command_queue: Any,
    output_queue: Any,
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
    method_name: str,
    method_params: dict[str, Any],
    profiles: list[ClientSpec],
    seed: int,
) -> None:
    """Own one CUDA device and process client jobs until shutdown."""
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    current_job = "startup"
    try:
        import random

        import numpy as np
        import torch

        from afl_vlm.data.afvlm_cm import AFVLMDataModule
        from afl_vlm.federation.client import FederatedClient
        from afl_vlm.methods.registry import create_method
        from afl_vlm.models.registry import create_model

        random.seed(seed)
        np.random.seed(seed)
        if not torch.cuda.is_available():
            raise RuntimeError(f"GPU worker {worker_id} cannot access CUDA device {device_id}")
        torch.cuda.set_device(device_id)
        torch.random.default_generator.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        worker_device = torch.device("cuda", device_id)
        requested_dtype = str(model_config.get("dtype", "fp16"))
        if requested_dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"GPU {device_id} does not support BF16; create an FP16 experiment profile"
            )
        output_queue.put(WorkerStatus(worker_id, f"loading model on visible GPU {device_id}"))
        worker_model_config = dict(model_config)
        worker_model_config["device_map"] = {"": device_id}
        worker_model_config["progress_bar"] = False
        model = create_model(worker_model_config["adapter"])
        model.load(worker_model_config)
        method = create_method(method_name, method_params)
        method.configure_model(model, profiles)
        data_module = AFVLMDataModule(dataset_config)
        initial_state = (
            state_to_device(model.snapshot_trainable(), "cpu") if worker_id == 0 else None
        )
        parameter_names = tuple(name for name, _ in model.named_federated_parameters())
        output_queue.put(WorkerReady(worker_id, device_id, initial_state, parameter_names))

        while True:
            command: WorkerCommand = command_queue.get()
            if isinstance(command, ShutdownWorker):
                return
            if isinstance(command, TrainJob):
                current_job = command.job_id
                event = command.event
                start_wall_time = time.time()
                output_queue.put(
                    WorkerStatus(worker_id, f"training {event.client_id} r{event.local_round}")
                )
                method.load_client_runtime_state(
                    nested_to_device(command.method_runtime_state, worker_device),
                    command.context,
                )
                command_job_id = command.job_id
                event_client_id = event.client_id
                event_id = event.event_id

                def progress(
                    step: int,
                    total: int,
                    loss: float,
                    job_id: str = command_job_id,
                    client_id: str = event_client_id,
                ) -> None:
                    output_queue.put(
                        WorkerProgress(
                            worker_id,
                            job_id,
                            client_id,
                            step,
                            total,
                            loss,
                        )
                    )

                request_counter = 0

                def fresh_state_provider(
                    job_id: str = command_job_id,
                    scheduled_event_id: int = event_id,
                ) -> tuple[LoRAState, int]:
                    nonlocal request_counter
                    request_counter += 1
                    request_id = f"{job_id}:fresh:{request_counter}"
                    output_queue.put(
                        FreshStateRequest(worker_id, scheduled_event_id, request_id)
                    )
                    response = command_queue.get()
                    if not isinstance(response, FreshStateResponse):
                        raise RuntimeError(
                            f"Expected FreshStateResponse for {request_id}, "
                            f"got {type(response).__name__}"
                        )
                    if response.request_id != request_id:
                        raise RuntimeError(
                            "Fresh-state response mismatch: "
                            f"{response.request_id} != {request_id}"
                        )
                    return state_to_device(response.state, worker_device), response.version

                samples = data_module.get_client_train_data(event.client_id)
                client = FederatedClient(event.client_id, event.task, event.dataset, samples)
                train_config = command.train_config
                train_config.progress_hook = progress
                update = client.train_prepared(
                    model=model,
                    method=method,
                    global_state=state_to_device(command.base_state, worker_device),
                    start_state=state_to_device(command.start_state, worker_device),
                    context=command.context,
                    train_config=train_config,
                    group_id=event.group_id,
                    group_size=command.group_size,
                    fresh_state_provider=fresh_state_provider
                    if method.capabilities.requires_fresh_global_during_local_training
                    else None,
                )
                update.metadata.update(
                    {
                        "worker_id": worker_id,
                        "visible_device_id": device_id,
                        "dispatch_wall_time": command.dispatch_wall_time,
                        "worker_start_wall_time": start_wall_time,
                        "worker_finish_wall_time": time.time(),
                    }
                )
                output_queue.put(
                    TrainCompleted(
                        worker_id,
                        event.event_id,
                        update_to_device(update, "cpu"),
                    )
                )
                continue
            if isinstance(command, EvaluationJob):
                current_job = command.job_id
                metrics = _evaluate_states(
                    model,
                    data_module,
                    {
                        key: state_to_device(state, worker_device)
                        for key, state in command.states.items()
                    },
                    command.split,
                    lambda message: output_queue.put(WorkerStatus(worker_id, message)),
                )
                output_queue.put(EvaluationCompleted(worker_id, command.job_id, metrics))
                continue
            raise TypeError(f"Unsupported worker command: {type(command).__name__}")
    except BaseException:
        output_queue.put(WorkerFailed(worker_id, current_job, traceback.format_exc()))


class ClientWorkerPool:
    """One spawned process and one resident model replica per configured GPU."""

    def __init__(
        self,
        devices: list[int],
        model_config: Mapping[str, Any],
        dataset_config: Mapping[str, Any],
        method_name: str,
        method_params: Mapping[str, Any],
        profiles: list[ClientSpec],
        seed: int,
        start_method: str = "spawn",
    ) -> None:
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("runtime.devices must contain unique GPU indices")
        self.devices = list(devices)
        self._context = mp.get_context(start_method)
        self._output = self._context.Queue()
        self._inputs = [self._context.Queue() for _ in devices]
        self._processes = []
        self._idle = set(range(len(devices)))
        self._busy: dict[int, str] = {}
        self._pending: deque[TrainJob | EvaluationJob] = deque()
        for worker_id, device_id in enumerate(devices):
            process = self._context.Process(
                target=_worker_main,
                args=(
                    worker_id,
                    device_id,
                    self._inputs[worker_id],
                    self._output,
                    dict(model_config),
                    dict(dataset_config),
                    method_name,
                    dict(method_params),
                    profiles,
                    seed,
                ),
                name=f"afvlm-gpu-{device_id}",
            )
            process.start()
            self._processes.append(process)

    @property
    def worker_count(self) -> int:
        return len(self._processes)

    def wait_until_ready(
        self,
        status_hook: Any | None = None,
        timeout_seconds: float = 1800.0,
    ) -> LoRAState:
        ready: dict[int, WorkerReady] = {}
        deadline = time.monotonic() + timeout_seconds
        while len(ready) < self.worker_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(set(range(self.worker_count)) - set(ready))
                raise TimeoutError(f"Timed out waiting for GPU workers to load: {missing}")
            try:
                message = self.receive(timeout=min(5.0, remaining))
            except queue.Empty:
                continue
            if isinstance(message, WorkerStatus):
                if status_hook:
                    status_hook(message)
                continue
            if isinstance(message, WorkerFailed):
                raise RuntimeError(
                    f"GPU worker {message.worker_id} failed during {message.job_id}:\n"
                    f"{message.traceback_text}"
                )
            if not isinstance(message, WorkerReady):
                raise RuntimeError(f"Unexpected startup message: {type(message).__name__}")
            ready[message.worker_id] = message
            if status_hook:
                status_hook(message)
        names = {item.parameter_names for item in ready.values()}
        if len(names) != 1:
            raise RuntimeError("GPU workers initialized incompatible federated parameter sets")
        initial_state = ready[0].initial_state
        if initial_state is None:
            raise RuntimeError("GPU worker 0 did not return the authoritative initial state")
        return initial_state

    def submit(self, job: TrainJob | EvaluationJob, *, priority: bool = False) -> None:
        if self._idle:
            self._dispatch(job)
        elif priority:
            self._pending.appendleft(job)
        else:
            self._pending.append(job)

    def _dispatch(self, job: TrainJob | EvaluationJob) -> None:
        worker_id = min(self._idle)
        self._idle.remove(worker_id)
        self._busy[worker_id] = job.job_id
        self._inputs[worker_id].put(job)

    def receive(self, timeout: float | None = None) -> WorkerMessage:
        poll_timeout = 5.0 if timeout is None else timeout
        while True:
            try:
                message: WorkerMessage = self._output.get(timeout=poll_timeout)
                break
            except queue.Empty as exc:
                failed = [
                    (worker_id, process.exitcode)
                    for worker_id, process in enumerate(self._processes)
                    if process.exitcode not in {None, 0}
                ]
                if failed:
                    raise RuntimeError(
                        f"GPU worker process exited unexpectedly: {failed}"
                    ) from exc
                if timeout is not None:
                    raise
        if isinstance(message, TrainCompleted | EvaluationCompleted):
            expected = self._busy.pop(message.worker_id, None)
            actual = (
                f"train:{message.event_id}"
                if isinstance(message, TrainCompleted)
                else message.job_id
            )
            if expected != actual:
                raise RuntimeError(
                    f"Worker {message.worker_id} completed {actual}, expected {expected}"
                )
            self._idle.add(message.worker_id)
            if self._pending:
                self._dispatch(self._pending.popleft())
        return message

    def receive_nowait(self) -> WorkerMessage:
        return self.receive(timeout=0.0)

    def respond_fresh(self, request: FreshStateRequest, state: LoRAState, version: int) -> None:
        self._inputs[request.worker_id].put(
            FreshStateResponse(request.request_id, state_to_device(state, "cpu"), version)
        )

    def shutdown(self) -> None:
        if self._busy or self._pending:
            raise RuntimeError("Cannot cleanly stop GPU workers while jobs remain")
        for worker_id in sorted(self._idle):
            self._inputs[worker_id].put(ShutdownWorker())
        for process in self._processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    def abort(self) -> None:
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=10)
