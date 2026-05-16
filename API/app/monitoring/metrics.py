import psutil
from prometheus_client import Counter, Gauge, Histogram, REGISTRY
from datetime import datetime

CPU_USAGE = Gauge("system_cpu_percent", "Uso de CPU en porcentaje")
RAM_USAGE = Gauge("system_ram_percent", "Uso de RAM en porcentaje")
RAM_USED_MB = Gauge("system_ram_used_mb", "RAM usada en MB")

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Número total de peticiones recibidas",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latencia de las peticiones en segundos",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

REQUEST_IN_PROGRESS = Gauge("api_requests_in_progress", "Peticiones en curso")


def collect_system_metrics():
    mem = psutil.virtual_memory()
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    RAM_USAGE.set(mem.percent)
    RAM_USED_MB.set(round(mem.used / 1024 / 1024, 1))


def current_snapshot() -> dict:
    collect_system_metrics()
    snapshot = {
        "system": {
            "cpu_percent": CPU_USAGE._value.get(),
            "ram_percent": RAM_USAGE._value.get(),
            "ram_used_mb": RAM_USED_MB._value.get(),
        },
        "requests": {
            "in_progress": REQUEST_IN_PROGRESS._value.get(),
            "total": {},
        },
        "latency": {},
    }
    for metric in REGISTRY.collect():
        if metric.name == "api_requests_total":
            for sample in metric.samples:
                key = "{method} {endpoint} {status_code}".format(**sample.labels)
                snapshot["requests"]["total"][key] = sample.value
        if metric.name == "api_request_latency_seconds":
            for sample in metric.samples:
                endpoint = sample.labels.get("endpoint", "unknown")
                if endpoint not in snapshot["latency"]:
                    snapshot["latency"][endpoint] = {"sum_s": 0.0, "count": 0, "avg_ms": 0.0}
                if sample.name.endswith("_sum"):
                    snapshot["latency"][endpoint]["sum_s"] = round(sample.value, 6)
                elif sample.name.endswith("_count"):
                    snapshot["latency"][endpoint]["count"] = int(sample.value)
    for ep, data in snapshot["latency"].items():
        if data["count"] > 0:
            data["avg_ms"] = round(data["sum_s"] / data["count"] * 1000, 3)
    return snapshot