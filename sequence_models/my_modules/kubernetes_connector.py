import statistics
import time
from kubernetes import client, config
from kubernetes.config import ConfigException
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from my_modules.printable_utils import logs


K8S_READY = False
APPS_V1 = None

def ensure_k8s_client():
    """
    Initialize K8s config + AppsV1Api exactly once (lazy).
    Safe for repeated calls.
    """
    global K8S_READY, APPS_V1
    if K8S_READY and APPS_V1 is not None:
        return APPS_V1

    try:
        # in-cluster
        config.load_incluster_config()
        logs("Loaded in-cluster Kubernetes config")
    except ConfigException:
        # local kubeconfig
        config.load_kube_config()
        logs("Loaded local kubeconfig")

    APPS_V1 = client.AppsV1Api()
    K8S_READY = True
    return APPS_V1


def scale_deployment(deployment_name: str, replicas: int, namespace="default"):
    """Scale number of replicas of a deployment."""
    apps_v1 = ensure_k8s_client()
    scale = {"spec": {"replicas": int(replicas)}}

    return apps_v1.patch_namespaced_deployment_scale(
        name=deployment_name,
        namespace=namespace,
        body=scale,
    )


def get_deployment_replicas(deployment_name: str, namespace="default"):
    """Return desired, current and available replicas for a deployment."""
    try:
        apps_v1 = ensure_k8s_client()
        deployment = apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
        )

        desired = deployment.spec.replicas
        current = deployment.status.replicas
        available = deployment.status.available_replicas

        return desired, current, available

    except Exception as e:
        logs(f"Kubernetes connection error: {e}")
        return None


def time_replica_creation(deployment_name: str, replicas: int, sleep_interval=1, namespace="default"):
    """Measure the time it takes replicas to become available."""
    start = datetime.now()
    scale_deployment(deployment_name, replicas, namespace)

    while True:
        desired_rep, current_rep, available_rep = get_deployment_replicas(deployment_name, namespace)
        logs(f"Replicas -> Desired:{desired_rep}, Current: {current_rep}, Available: {available_rep}")

        if available_rep == desired_rep:
            break
        time.sleep(sleep_interval)

    end = datetime.now()
    elapsed = (end - start).total_seconds()
    logs(f"Replica created: {start} -> {end} (took {elapsed:.2f} seconds)")
    return elapsed

AUTOSCALE_STATE = defaultdict(lambda: {
    "last_upscaled_at": None,
    "last_downscaled_at": None,
})


def choose_signal(predicted_values: list, strategy: str):
    if strategy == "max":
        return float(max(predicted_values))
    if strategy == "min":
        return float(min(predicted_values))
    if strategy == "next":
        return float(predicted_values[0])
    if strategy == "avg":
        return float(sum(predicted_values) / len(predicted_values))
    if strategy == "median":
        return float(statistics.median(predicted_values))
    raise ValueError(f"Unknown strategy: {strategy}")


def hpa_like_autoscale(
    scale_signal: float,
    target_cpu_percent: int,
    desired_rep: int,
    min_replicas: int,
    max_replicas: int,
    upscale_replica_limit: int = 1,
    downscale_replica_limit: int = 1,
):
    ratio = scale_signal / target_cpu_percent
    raw_new_replicas = int(np.ceil(desired_rep * ratio))
    min_max_new_replicas = max(min_replicas, min(raw_new_replicas, max_replicas))

    if min_max_new_replicas > desired_rep:
        new_replicas = min(desired_rep + upscale_replica_limit, min_max_new_replicas)
    elif min_max_new_replicas < desired_rep:
        new_replicas = max(desired_rep - downscale_replica_limit, min_max_new_replicas)
    else:
        new_replicas = desired_rep

    delta = new_replicas - desired_rep
    return new_replicas, delta

def kuber_replica_autoscale(
    deployment_name: str,
    predicted_cpu_percentages,
    namespace: str = "default",
    *,
    current_cpu_percent: float | None = None,
    target_cpu_percent: int = 60,
    strategy: str = "max",
    min_replicas: int = 1,
    max_replicas: int = 6,
    upscale_cooldown_seconds: int = 0,
    downscale_cooldown_seconds: int = 300,
    downscale_after_upscale_delay_seconds: int = 300,
    upscale_threshold: int = 60,
    downscale_threshold: int = 30,
    upscale_replica_limit: int = 1,
    downscale_replica_limit: int = 1,
    **_,
):
    # validate
    try:
        if hasattr(predicted_cpu_percentages, "tolist"):
            predicted_cpu_percentages = predicted_cpu_percentages.tolist()
        predicted_cpu_percentages = [float(p) for p in predicted_cpu_percentages]
    except Exception as e:
        logs(f"{deployment_name}: invalid predicted_cpu_percentages ({predicted_cpu_percentages}): {e}")
        return None

    if not predicted_cpu_percentages:
        logs(f"{deployment_name}: empty predicted_cpu_percentages")
        return None

    try:
        if current_cpu_percent is not None:
            current_cpu_percent = float(current_cpu_percent)
    except Exception as e:
        logs(f"{deployment_name}: invalid current_cpu_percent ({current_cpu_percent}): {e}")
        current_cpu_percent = None

    if target_cpu_percent <= 0:
        logs(f"{deployment_name}: invalid target_cpu_percent ({target_cpu_percent})")
        return None

    desired_rep, _, _ = get_deployment_replicas(deployment_name, namespace)
    if desired_rep is None:
        logs(f"{deployment_name}: could not read desired replicas")
        return None

    # choose signal
    predicted_signal = choose_signal(predicted_cpu_percentages, strategy=strategy)

    # signals for scaling
    if current_cpu_percent is not None:
        up_signal = max(current_cpu_percent, predicted_signal)
        if current_cpu_percent == predicted_signal:
            signal_source = "current=predicted"
        elif up_signal == predicted_signal:
            signal_source = "predicted"
        else:
            signal_source = "current"
    else:
        up_signal = predicted_signal
        signal_source = "predicted"
    
    # for logging
    pred_print = [f"{p:.2f}" for p in predicted_cpu_percentages]
    curr_print = f"{current_cpu_percent:.2f}" if current_cpu_percent is not None else "None"    
    # downscale only if both current and predicted signals are below the threshold (if current is available)
    down_current_ok = (current_cpu_percent is not None and current_cpu_percent <= downscale_threshold)
    down_pred_ok = predicted_signal <= downscale_threshold

    # decide direction and scale signal value
    if up_signal >= upscale_threshold:
        direction = "up"
        scale_signal = up_signal
    elif down_current_ok and down_pred_ok:
        direction = "down"
        scale_signal = current_cpu_percent
    else:
        logs(
            f"{scaling_info(deployment_name, pred_print, predicted_signal, signal_source, curr_print, up_signal, strategy, target_cpu_percent, upscale_threshold, downscale_threshold)} "
            f"-> signal in norm -> No change"
        )
        return None
    
    # compute new replicas with HPA-like logic
    new_replicas, delta = hpa_like_autoscale(
        scale_signal=scale_signal,
        target_cpu_percent=target_cpu_percent,
        desired_rep=desired_rep,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        upscale_replica_limit=upscale_replica_limit,
        downscale_replica_limit=downscale_replica_limit,
    )
    
    # scale but replica count unchanged -> skip (but log it)
    if delta == 0:
        logs(
            f"{scaling_info(deployment_name, pred_print, predicted_signal, signal_source, curr_print, scale_signal, strategy, target_cpu_percent, upscale_threshold, downscale_threshold)} "
            f"-> {direction} signaled but replica count unchanged"
        )
        return None

    # consistency check
    if direction == "up" and delta < 0:
        logs(f"{deployment_name}: inconsistent upscale decision ({desired_rep} -> {new_replicas}) -> skip")
        return None
    if direction == "down" and delta > 0:
        logs(f"{deployment_name}: inconsistent downscale decision ({desired_rep} -> {new_replicas}) -> skip")
        return None

    # cooldown check
    st = AUTOSCALE_STATE.setdefault(
        deployment_name,
        {"last_upscaled_at": None, "last_downscaled_at": None}
    )
    now = datetime.now()
    if delta > 0:
        if st["last_upscaled_at"] is not None:
            elapsed = (now - st["last_upscaled_at"]).total_seconds()
            if elapsed < upscale_cooldown_seconds:
                wait_left = int(upscale_cooldown_seconds - elapsed)
                logs(f"{deployment_name} within upscale cooldown ({wait_left}s left) -> skip")
                return None
    else:
        # block downscale shortly after previous downscale
        if st["last_downscaled_at"] is not None:
            elapsed = (now - st["last_downscaled_at"]).total_seconds()
            if elapsed < downscale_cooldown_seconds:
                wait_left = int(downscale_cooldown_seconds - elapsed)
                logs(f"{deployment_name} within downscale cooldown after previous downscale ({wait_left}s left) -> skip")
                return None

        # block downscale shortly after previous upscale
        if st["last_upscaled_at"] is not None:
            elapsed = (now - st["last_upscaled_at"]).total_seconds()
            if elapsed < downscale_after_upscale_delay_seconds:
                wait_left = int(downscale_after_upscale_delay_seconds - elapsed)
                logs(f"{deployment_name} waiting before downscale after previous upscale ({wait_left}s left) -> skip")
                return None

    # scale
    scale_deployment(deployment_name, new_replicas, namespace)
    # update state
    if delta > 0:
        st["last_upscaled_at"] = now
        scale_direction = "up"
    else:
        st["last_downscaled_at"] = now
        scale_direction = "down"

    logs(
        f"{scaling_info(deployment_name, pred_print, predicted_signal, signal_source, curr_print, scale_signal, strategy, target_cpu_percent, upscale_threshold, downscale_threshold)} "
        f"-> scaling {scale_direction} {desired_rep} -> {new_replicas}"
    )

    return new_replicas


def scaling_info(deployment_name, pred_print, predicted_signal, signal_source, curr_print, scale_signal, strategy, target_cpu_percent, upscale_threshold, downscale_threshold):
    return (
        f"{deployment_name}: "
        f"pred={pred_print}, "
        f"strategy='{strategy}', "
        f"predicted_signal={predicted_signal:.2f}%, "
        f"signal_source={signal_source}, "
        f"current={curr_print}%, "
        f"target_cpu={target_cpu_percent}%, "
        f"upscale_threshold={upscale_threshold}%, "
        f"downscale_threshold={downscale_threshold}%, "
        f"scale_signal={scale_signal:.2f}%"
    )