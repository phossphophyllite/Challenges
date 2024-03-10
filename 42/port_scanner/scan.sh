#!/bin/bash

IP_VERSION=6
if [[ "$1" == "ipv4" ]]; then
    IP_VERSION=4
fi

if [[ $IP_VERSION -eq 6 ]]; then
    IP=$(ip -6 addr show scope global | grep "inet6" | awk '{print $2}' | cut -d'/' -f1 | head -n1)
else 
    IP=$(ip -4 addr show scope global | grep "inet" | awk '{print $2}' | cut -d'/' -f1 | head -n1)
fi
if [[ "$2" != "" ]]; then
	IP=$2
fi

if [[ -z "$IP" ]]; then
    echo "Failed to determine IP address."
    exit 1
fi

START_PORT=1
END_PORT=65535


RUST_BINARY_PATH="./target/release/port_scanner"

echo "Using IP address: $IP"

"$RUST_BINARY_PATH" "$IP" "$START_PORT" "$END_PORT"

