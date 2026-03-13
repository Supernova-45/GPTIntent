#!/usr/bin/env bash
# Script to use linux namespaces to collect data in parallel

set -euo pipefail

count=${2:-5}
action=${1:-}

if [[ $EUID -ne 0 ]]; then
  echo "Must run as root" >&2
  exit 1
fi

uplink=$(ip route show default 2>/dev/null | awk 'NR==1 {print $5}')

add_ns() {
  local idx=$1

  echo "Creating namespace wlns${idx}"

  ip netns add "wlns${idx}"
  ip link add "veth${idx}h" type veth peer name "veth${idx}n"
  ip link set "veth${idx}n" netns "wlns${idx}"
  ip addr add "10.200.${idx}.1/30" dev "veth${idx}h"
  ip link set "veth${idx}h" up
  ip netns exec "wlns${idx}" ip addr add "10.200.${idx}.2/30" dev "veth${idx}n"
  ip netns exec "wlns${idx}" ip link set "veth${idx}n" up
  ip netns exec "wlns${idx}" ip link set lo up
  ip netns exec "wlns${idx}" ip route add default via "10.200.${idx}.1"

  iptables -t nat -C POSTROUTING -s "10.200.${idx}.0/30" -o "$uplink" -j MASQUERADE 2>/dev/null \
    || iptables -t nat -A POSTROUTING -s "10.200.${idx}.0/30" -o "$uplink" -j MASQUERADE
  iptables -C FORWARD -i "$uplink" -o "veth${idx}h" -j ACCEPT 2>/dev/null \
    || iptables -A FORWARD -i "$uplink" -o "veth${idx}h" -j ACCEPT
  iptables -C FORWARD -i "veth${idx}h" -o "$uplink" -j ACCEPT 2>/dev/null \
    || iptables -A FORWARD -i "veth${idx}h" -o "$uplink" -j ACCEPT

  mkdir -p "/etc/netns/wlns${idx}"
  echo "nameserver 1.1.1.1" > "/etc/netns/wlns${idx}/resolv.conf"
  echo "nameserver 8.8.8.8" >> "/etc/netns/wlns${idx}/resolv.conf"
}

del_ns() {
  local idx=$1

  echo "Removing namespace wlns${idx}"

  iptables -t nat -D POSTROUTING -s "10.200.${idx}.0/30" -o "$uplink" -j MASQUERADE 2>/dev/null || true
  iptables -D FORWARD -i "$uplink" -o "veth${idx}h" -j ACCEPT 2>/dev/null || true
  iptables -D FORWARD -i "veth${idx}h" -o "$uplink" -j ACCEPT 2>/dev/null || true

  ip netns del "wlns${idx}" 2>/dev/null || true
  ip link del "veth${idx}h" 2>/dev/null || true
  rm -rf "/etc/netns/wlns${idx}" 2>/dev/null || true
}

case "$action" in
  create)
    sysctl -w net.ipv4.ip_forward=1 >/dev/null
    for i in $(seq 0 $((count-1))); do
      add_ns "$i"
    done
    ;;
  destroy)
    for i in $(seq 0 $((count-1))); do
      del_ns "$i"
    done
    ;;
esac