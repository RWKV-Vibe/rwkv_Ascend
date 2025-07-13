#!/bin/bash
echo "[ascend910b] Generating wkv7grad_d842dbcd0d2ce4b7316f258aae6918a4 ..."
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(opc $1 --main_func=wkv7grad --input_param=/root/Main/cann-ops-master/build/binary/ascend910b/gen/wkv7grad_d842dbcd0d2ce4b7316f258aae6918a4_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/wkv7grad_d842dbcd0d2ce4b7316f258aae6918a4.json ; then
  echo "$2/wkv7grad_d842dbcd0d2ce4b7316f258aae6918a4.json not generated!"
  exit 1
fi

if ! test -f $2/wkv7grad_d842dbcd0d2ce4b7316f258aae6918a4.o ; then
  echo "$2/wkv7grad_d842dbcd0d2ce4b7316f258aae6918a4.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating wkv7grad_d842dbcd0d2ce4b7316f258aae6918a4 Done"
