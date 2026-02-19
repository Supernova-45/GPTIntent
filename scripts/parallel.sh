#!/usr/bin/env bash
# Helper to spin up lightweight network namespaces with veth pairs
# so multiple collectors can run in parallel with isolated TLS captures.

set -euo pipefail

count=${2:-}
action=${1:-}

if [[ -z "$action" ]]; then
  echo "Usage: sudo $0 <create|destroy> <count>" >&2
  exit 1
fi

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root" >&2
  exit 1
fi

uplink=$(ip route show default 2>/dev/null | awk 'NR==1 {print $5}')
if [[ -z "$uplink" ]]; then
  echo "Could not determine uplink interface" >&2
  exit 1
fi

enable_ip_forward() {
  if [[ "$(sysctl -n net.ipv4.ip_forward)" -ne 1 ]]; then
    sysctl -w net.ipv4.ip_forward=1 >/dev/null
  fi
}

add_ns() {
  local idx=$1
  local ns="wlns${idx}"
  local host_if="veth${idx}h"
  local ns_if="veth${idx}n"
  local subnet="10.200.${idx}.0/30"
  local host_ip="10.200.${idx}.1"
  local ns_ip="10.200.${idx}.2"
  local ns_conf_dir="/etc/netns/${ns}"

  echo "Creating namespace $ns with $host_if <-> $ns_if ($subnet)"

  ip netns add "$ns"
  ip link add "$host_if" type veth peer name "$ns_if"
  ip link set "$ns_if" netns "$ns"

  ip addr add "${host_ip}/30" dev "$host_if"
  ip link set "$host_if" up

  ip netns exec "$ns" ip addr add "${ns_ip}/30" dev "$ns_if"
  ip netns exec "$ns" ip link set "$ns_if" up
  ip netns exec "$ns" ip link set lo up
  ip netns exec "$ns" ip route add default via "$host_ip"

  iptables -t nat -C POSTROUTING -s "$subnet" -o "$uplink" -j MASQUERADE 2>/dev/null \
    || iptables -t nat -A POSTROUTING -s "$subnet" -o "$uplink" -j MASQUERADE
  iptables -C FORWARD -i "$uplink" -o "$host_if" -j ACCEPT 2>/dev/null \
    || iptables -A FORWARD -i "$uplink" -o "$host_if" -j ACCEPT
  iptables -C FORWARD -i "$host_if" -o "$uplink" -j ACCEPT 2>/dev/null \
    || iptables -A FORWARD -i "$host_if" -o "$uplink" -j ACCEPT

  # Configure DNS for the namespace (use public resolvers to avoid localhost stubs)
  mkdir -p "$ns_conf_dir"
  cat > "$ns_conf_dir/resolv.conf" <<'EOF'
nameserver 1.1.1.1
nameserver 8.8.8.8
EOF
}

del_ns() {
  local idx=$1
  local ns="wlns${idx}"
  local host_if="veth${idx}h"
  local subnet="10.200.${idx}.0/30"
  local ns_conf_dir="/etc/netns/${ns}"

  echo "Removing namespace $ns"

  iptables -t nat -D POSTROUTING -s "$subnet" -o "$uplink" -j MASQUERADE 2>/dev/null || true
  iptables -D FORWARD -i "$uplink" -o "$host_if" -j ACCEPT 2>/dev/null || true
  iptables -D FORWARD -i "$host_if" -o "$uplink" -j ACCEPT 2>/dev/null || true

  ip netns del "$ns" 2>/dev/null || true
  ip link del "$host_if" 2>/dev/null || true
  rm -rf "$ns_conf_dir" 2>/dev/null || true
}

case "$action" in
  create)
    if [[ -z "$count" ]]; then
      echo "Please specify a count" >&2
      exit 1
    fi
    enable_ip_forward
    for i in $(seq 0 $((count-1))); do
      add_ns "$i"
    done
    ;;
  destroy)
    if [[ -z "$count" ]]; then
      echo "Please specify a count" >&2
      exit 1
    fi
    for i in $(seq 0 $((count-1))); do
      del_ns "$i"
    done
    ;;
  *)
    echo "Unknown action: $action" >&2
    exit 1
    ;;
esac

echo "Done."
