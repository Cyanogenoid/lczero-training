#!/usr/bin/env bash

protoc --proto_path=libs/lczero-common --python_out=tf libs/lczero-common/proto/net.proto
protoc --proto_path=libs/lczero-common --python_out=tf libs/lczero-common/proto/chunk.proto
protoc --proto_path=libs/lczero-common --python_out=pytorch libs/lczero-common/proto/net.proto
protoc --proto_path=libs/lczero-common --python_out=pytorch libs/lczero-common/proto/chunk.proto
touch tf/proto/__init__.py
touch pytorch/proto/__init__.py
