#!/bin/bash
# corpus swarm worker startup (Debian 12, spot). Parameters via metadata:
#   seed_count, shard (k/N), extent, rays, gcs_prefix, start (optional, default 0)
exec > /var/log/corpus_startup.log 2>&1
set -x
export HOME=/root   # metadata-script-runner leaves $HOME unset; Dr.Jit
                     # needs it for its ~/.drjit cache dir (jit_init crash)
trap 'gcloud storage cp /var/log/corpus_startup.log "$(M gcs_prefix)/logs/startup_$(hostname)_$(date +%s).log" || true' EXIT
M() { curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
apt-get update -qq
apt-get install -y -qq python3-pip libllvm16 >/dev/null
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-16.so
mkdir -p /opt/corpus && cd /opt/corpus
for i in 1 2 3; do
  gcloud storage cp gs://urban-water-detection-sweep/code/sarsim.tgz . && break
  sleep 15
done
tar xzf sarsim.tgz
python3 -m pip install --break-system-packages --quiet numpy scipy tqdm mitsuba
export PYTHONPATH=/opt/corpus/sarsim/src
COUNT=$(M seed_count); SHARD=$(M shard); EXTENT=$(M extent); RAYS=$(M rays); PREFIX=$(M gcs_prefix)
START=$(M start); START=${START:-0}
python3 sarsim/scripts/render_corpus.py --start "$START" --count "$COUNT" --shard "$SHARD" --rotate \
  --extent "$EXTENT" --rays "$RAYS" --gcs "$PREFIX" > /var/log/corpus.log 2>&1
gcloud storage cp /var/log/corpus.log "$PREFIX/logs/$(hostname)_$(date +%s).log"
poweroff
