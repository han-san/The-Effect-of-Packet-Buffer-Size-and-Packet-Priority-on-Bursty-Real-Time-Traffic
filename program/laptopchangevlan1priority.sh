#! /bin/sh

# Changes pi vlan 1 priority.

if test -z $1; then
  echo "Provide the priority you want to set vlan 1 to"
  exit 1
fi

ip link delete vlan1
ip link add link enp0s31f6 name vlan1 type vlan id 1 egress-qos-map "0:${1}"
ip link set dev vlan1 up
ip a add 192.168.1.2/24 dev vlan1
