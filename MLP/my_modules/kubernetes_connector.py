import time
import statistics

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


def choose_signal(predicted_values: list, strategy: str):
    if strategy == "max":
        return int(max(predicted_values))
    if strategy == "min":
        return int(min(predicted_values))
    if strategy == "next":
        return int(predicted_values[0])
    if strategy == "avg":
        return int(sum(predicted_values) / len(predicted_values))
    if strategy == "median":
        return int(statistics.median(predicted_values))    
    raise ValueError(f"Unknown strategy: {strategy}")

AUTOSCALE_STATE = defaultdict(lambda: {
    "last_upscaled_at": None,
    "last_downscaled_at": None,
})


# def get_scaling(
#     predictions,
#     current_class, 
#     current_replicas,
#     UP_CLASS = 3,
#     DOWN_CLASS = 0,
#     min_reps=1, 
#     max_reps=6
# ):
#     UP_STEP = 1
#     UP_SCALE_TREND = 1
#     UP_RISK_CLASS = 4
#     UP_RISK_TREND = 1

#     DOWN_STEP = 1
#     DOWN_SCALE_TREND = 4

#     preds = list(map(int, predictions))
#     current_class = int(current_class)
#     current       = int(current_replicas)

#     high_count = sum(c >= UP_CLASS for c in preds)                 # >=60%
#     low_count  = sum(c <= DOWN_CLASS for c in preds)               # <=40%
#     risk_count = sum(c >= UP_RISK_CLASS for c in preds)            # >=80%

#     if (current_class >= UP_CLASS) or (high_count >= UP_SCALE_TREND) or (risk_count >= UP_RISK_TREND):
#         delta = UP_STEP
#     elif (current_class <= DOWN_CLASS) and (low_count >= DOWN_SCALE_TREND):
#         delta = -DOWN_STEP
#     else:
#         delta = 0

#     new = max(min_reps, min(current + delta, max_reps))
#     return new, new - current

def kuber_replica_class_autoscale(
    deployment_name: str,
    predicted_cpu_classes,
    scaling_config,
    current_class,
    namespace: str = "default",
    upscale_cooldown_seconds: int = 0,
    downscale_cooldown_seconds: int = 300,
    downscale_after_upscale_delay_seconds: int = 300,
    strategy="max",
    UP_CLASS=3,
    DOWN_CLASS=0,
    TARGET_CLASS=2,
    require_ready_before_next: bool = False,
    **_,
):
    try:
        if hasattr(predicted_cpu_classes, "tolist"):
            predicted_cpu_classes = predicted_cpu_classes.tolist()
        predicted_cpu_classes = list(map(int, predicted_cpu_classes))
    except Exception as e:
        logs(f"{deployment_name}: invalid predicted_cpu_classes ({predicted_cpu_classes}): {e}")
        return None

    if not predicted_cpu_classes:
        logs(f"{deployment_name}: empty predicted_cpu_classes")
        return None

    try:
        if current_class is not None:
            current_class = int(current_class)
    except Exception as e:
        logs(f"{deployment_name}: invalid current_class ({current_class}): {e}")
        current_class = None

    # uncomment if in use
    # if TARGET_CLASS <= 0:
    #     logs(f"{deployment_name}: invalid TARGET_CLASS ({TARGET_CLASS})")
    #     return None

    svc = scaling_config.get("services", {}).get(deployment_name)
    if not svc:
        logs(f"{deployment_name}: missing scaling_config for service -> skip")
        return None

    min_replicas = int(svc.get("min_pods", 1))
    max_replicas = int(svc.get("max_pods", 6))

    replica_res = get_deployment_replicas(deployment_name, namespace)
    if replica_res is None:
        logs(f"{deployment_name}: cannot read replicas -> skip")
        return None
    desired_rep, _, available_rep = replica_res

    # signals
    predicted_signal = choose_signal(predicted_cpu_classes, strategy=strategy)

    if current_class is not None:
        scale_signal = max(current_class, predicted_signal)
        if current_class == predicted_signal:
            signal_source = "current&predicted"
        elif scale_signal == predicted_signal:
            signal_source = "predicted"
        else:
            signal_source = "current"
    else:
        scale_signal = predicted_signal
        signal_source = "predicted"

    pred_print = [str(p) for p in predicted_cpu_classes]
    curr_print = str(current_class) if current_class is not None else "None"


    # decision
    if scale_signal >= UP_CLASS:
        direction = "up"
        if scale_signal >= UP_CLASS:
            raw_replicas = 1
        else:
            raw_replicas = 2
    elif scale_signal <= DOWN_CLASS:
        direction = "down"
        raw_replicas = -1
    else:
        logs(
            f"{scaling_info(deployment_name, pred_print, predicted_signal, signal_source, curr_print, scale_signal, strategy, TARGET_CLASS, UP_CLASS, DOWN_CLASS)} "
            f"-> signal in norm -> No change"
        )
        return None

    # rate limit
    new_replicas = max(min_replicas, min(desired_rep + raw_replicas, max_replicas))
    delta = new_replicas - desired_rep

    if delta == 0:
        logs(
            f"{scaling_info(deployment_name, pred_print, predicted_signal, signal_source, curr_print, scale_signal, strategy, TARGET_CLASS, UP_CLASS, DOWN_CLASS)} "
            f"-> {direction} signaled but replica count unchanged"
        )
        return None

    # consistency guard
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

    # if require_ready_before_next and delta < 0 and desired_rep != available_rep:
    #     logs(f"{deployment_name} not stable enough for downscale (Desired={desired_rep}, Available={available_rep}) -> skip")
    #     return None

    # apply
    scale_deployment(deployment_name, new_replicas, namespace)

    # update state
    if delta > 0:
        st["last_upscaled_at"] = now
        scale_direction = "up"
    else:
        st["last_downscaled_at"] = now
        scale_direction = "down"

    logs(
        f"{scaling_info(deployment_name, pred_print, predicted_signal, signal_source, curr_print, scale_signal, strategy, TARGET_CLASS, UP_CLASS, DOWN_CLASS)} "
        f"-> scaling {scale_direction} {desired_rep} -> {new_replicas}"
    )
    return new_replicas


def scaling_info(deployment_name, pred_print, predicted_signal, signal_source, curr_print, scale_signal, strategy, TARGET_CLASS, UP_CLASS, DOWN_CLASS):
    return (
        f"{deployment_name}: "
        f"pred={pred_print}, "
        f"strategy='{strategy}', "
        f"predicted_signal={predicted_signal}, "
        f"signal_source={signal_source}, "
        f"current={curr_print}, "
        # f"target_class={TARGET_CLASS}, "
        f"UP_CLASS={UP_CLASS} "
        f"DOWN_CLASS={DOWN_CLASS}, "
        f"scale_signal={scale_signal:.2f}%"
    )





# def kuber_replica_class_autoscale(
#         deployment_name: str,
#         predicted_cpu_classes,
#         namespace: str = "default",
#         *,
#         class_to_delta = {0:-1, 1:0, 2:+1, 3:+2, 4:+3},
#         strategy: str = "avg",
#         min_replicas: int = 1,
#         max_replicas: int = 200,
#         max_step_per_cycle: int = 2,
#         up_persistence: int = 2,
#         down_persistence: int = 3,
#         cooldown_seconds: int = 180,
#         require_ready_before_next: bool = True,
#         **_
#     ):
#     # Ensure list
#     try:
#         if hasattr(predicted_cpu_classes, "tolist"):
#             predicted_cpu_classes = predicted_cpu_classes.tolist()
#         predicted_cpu_classes = list(map(int, predicted_cpu_classes))
#     except Exception as e:
#         logs(f"{deployment_name}: invalid predicted_cpu_classes ({predicted_cpu_classes}): {e}")
#         return None

#     # Cooldown
#     st = AUTOSCALE_STATE[deployment_name]
#     now = datetime.now()
#     if st["last_scaled_at"] is not None and (now - st["last_scaled_at"]) < timedelta(seconds=cooldown_seconds):
#         wait_left = int(cooldown_seconds - (now - st["last_scaled_at"]).total_seconds())
#         logs(f"{deployment_name} within cooldown ({wait_left}s left) -> skip")
#         return None

#     # Collapse horizon to one signal
#     signal_class = choose_signal(predicted_cpu_classes, strategy=strategy)
#     delta = int(class_to_delta.get(signal_class, 0))

#     # Cluster state
#     replica_res = get_deployment_replicas(deployment_name, namespace)
#     if replica_res is None:
#         logs(f"{deployment_name}: cannot read replicas -> skip")
#         return None
#     desired_rep, _, available_rep = replica_res

#     # Do-not-stack rule with pending stuck solution
#     if require_ready_before_next and desired_rep != available_rep and delta > 0:
#         logs(f"{deployment_name} still scaling up (Desired={desired_rep}, Available={available_rep}) -> skip")
#         return None

#     # Update streaks
#     if st["last_signal"] == signal_class:
#         st["up_streak"]   = st["up_streak"] + 1 if delta > 0 else 0
#         st["down_streak"] = st["down_streak"] + 1 if delta < 0 else 0
#     else:
#         st["last_signal"] = signal_class
#         st["up_streak"]   = 1 if delta > 0 else 0
#         st["down_streak"] = 1 if delta < 0 else 0

#     # Persistence gates
#     if delta > 0 and st["up_streak"] < up_persistence:
#         logs(f"{deployment_name}: up-signal class={signal_class}, streak {st['up_streak']}/{up_persistence} -> hold")
#         return None
#     if delta < 0 and st["down_streak"] < down_persistence:
#         logs(f"{deployment_name}: down-signal class={signal_class}, streak {st['down_streak']}/{down_persistence} -> hold")
#         return None

#     # Rate limit
#     applied_delta = max(-max_step_per_cycle, min(delta, max_step_per_cycle))
#     new_replicas = max(min_replicas, min(desired_rep + applied_delta, max_replicas))

#     if applied_delta == 0 or new_replicas == desired_rep:
#         logs(f"{deployment_name}: Pred={predicted_cpu_classes}, class={signal_class}) -> No change")
#         return None

#     # Apply
#     scale_deployment(deployment_name, new_replicas, namespace)
#     st["last_scaled_at"] = now
#     st["up_streak"] = 0
#     st["down_streak"] = 0

#     logs(f"{deployment_name}: Pred={predicted_cpu_classes}, class={signal_class}, "
#          f"delta={delta}, applied_delta={applied_delta}) -> {desired_rep} → {new_replicas}")
#     return new_replicas