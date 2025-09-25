source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export VTUNE_LIB=/opt/intel/oneapi/vtune/latest/lib64
echo "VTUNE_LIB after export: "
export LD_LIBRARY_PATH=""
echo "LD_LIBRARY_PATH inside script: "
