#!/bin/bash
# Launch N corpus-render spot VMs in us-east1. Usage:
#   launch_swarm.sh N_VMS SEED_COUNT EXTENT RAYS PREFIX
set -e
N=${1:-6}; COUNT=${2:-3000}; EXTENT=${3:-320}; RAYS=${4:-384}
PREFIX=${5:-gs://urban-water-detection-sweep/synthetic_corpus/v1}
SCRATCH="$(dirname "$0")"
ZONES=(us-east1-b us-east1-c us-east1-d)   # spread for stock/preemption
for k in $(seq 0 $((N-1))); do
  Z=${ZONES[$((k % ${#ZONES[@]}))]}
  gcloud compute instances create "corpus-$k" \
    --zone="$Z" --machine-type=n2-standard-16 \
    --provisioning-model=SPOT --instance-termination-action=DELETE \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=30GB \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write \
    --metadata-from-file=startup-script="$SCRATCH/corpus_startup.sh" \
    --metadata=seed_count="$COUNT",shard="$k/$N",extent="$EXTENT",rays="$RAYS",gcs_prefix="$PREFIX" \
    2>&1 | tail -1 &
done
wait
echo "launched $N VMs -> $PREFIX"
