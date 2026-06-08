"""Serial Docker build script — Phase 10.

Builds all 7 service images one at a time so the 4 Mbps link isn't
saturated by parallel pip downloads. CPU-base services are built from
docker-compose.yml only; GPU services (retrieval, rag) use both compose
files for the CUDA overlay.

Usage:
    python scripts/build_docker_all.py            # full build
    python scripts/build_docker_all.py --push     # also docker push (future)
"""

import subprocess
import sys
import os
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SERVICES_CPU = ["preprocessing", "indexing", "refinement", "gateway", "ui"]
SERVICES_GPU = ["retrieval", "rag"]
ALL_SERVICES = SERVICES_CPU + SERVICES_GPU

BASE_COMPOSE = "docker-compose.yml"
GPU_COMPOSE = "docker-compose.gpu.yml"

DOCKER_EXE = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"


def build_cpu(service: str) -> bool:
    print(f"\n{'='*60}")
    print(f"BUILD CPU: {service}")
    print(f"{'='*60}")
    cmd = [
        DOCKER_EXE, "compose", "-f", BASE_COMPOSE,
        "build", service
    ]
    env = os.environ.copy()
    env["COMPOSE_IGNORE_ORPHANS"] = "True"
    env["DOCKER_BUILDKIT"] = "1"
    env["COMPOSE_DOCKER_CLI_BUILD"] = "1"
    # Ensure PATH is set for subprocess (Windows quirk)
    env["PATH"] = os.environ.get("PATH", "") + os.pathsep + os.path.dirname(DOCKER_EXE)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, shell=True)
    elapsed = time.time() - t0
    ok = r.returncode == 0
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {service} — {elapsed:.1f}s")
    return ok


def build_gpu(service: str) -> bool:
    print(f"\n{'='*60}")
    print(f"BUILD GPU: {service}")
    print(f"{'='*60}")
    cmd = [
        DOCKER_EXE, "compose", "-f", BASE_COMPOSE, "-f", GPU_COMPOSE,
        "build", service
    ]
    env = os.environ.copy()
    env["COMPOSE_IGNORE_ORPHANS"] = "True"
    env["DOCKER_BUILDKIT"] = "1"
    env["COMPOSE_DOCKER_CLI_BUILD"] = "1"
    env["PATH"] = os.environ.get("PATH", "") + os.pathsep + os.path.dirname(DOCKER_EXE)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, shell=True)
    elapsed = time.time() - t0
    ok = r.returncode == 0
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {service} — {elapsed:.1f}s")
    return ok


def main():
    push = "--push" in sys.argv
    failures = []

    print("=== IR Project — Docker Build All ===")
    print(f"CPU services: {SERVICES_CPU}")
    print(f"GPU services: {SERVICES_GPU}")
    print()

    for svc in ALL_SERVICES:
        for attempt in range(1, 4):  # 3 retries per service
            if svc in SERVICES_GPU:
                ok = build_gpu(svc)
            else:
                ok = build_cpu(svc)
            if ok:
                break
            if attempt < 3:
                print(f"--- {svc} failed (attempt {attempt}/3), retrying entire service build in 60s ---")
                time.sleep(60)
        else:
            failures.append(svc)

    print(f"\n{'='*60}")
    print("BUILD SUMMARY")
    print(f"{'='*60}")
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        sys.exit(1)
    else:
        print("ALL 7 IMAGES BUILT SUCCESSFULLY")

    if push:
        print("Push flag set — pushing images...")
        for svc in ALL_SERVICES:
            tag = f"ir-project/{svc}:latest"
            print(f"  docker push {tag}")
            subprocess.run([DOCKER_EXE, "image", "push", tag], cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
