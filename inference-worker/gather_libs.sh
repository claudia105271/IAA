#!/bin/bash
mkdir -p inference-worker/libs
cp /usr/lib/x86_64-linux-gnu/libPVROCL*.so inference-worker/libs/
cp /usr/lib/x86_64-linux-gnu/libPVROCL.so* inference-worker/libs/
cp /usr/lib/x86_64-linux-gnu/libsrv_um.so* inference-worker/libs/
cp /usr/lib/x86_64-linux-gnu/libufwriter.so* inference-worker/libs/
cp /usr/lib/x86_64-linux-gnu/libusc.so* inference-worker/libs/
