# SPDX-License-Identifier: GPL-3.0-only
# cmake -DF1=<file> [-DF2=<file> ... -DF4=<file>] -DOUT=<list> -P sha256_list.cmake: one line "<sha256> <file name>" per file
# (the testing build's list of WrongGPU.exe and the launcher suite's canaries, tests/launcher_build_facts.py
# verified_canaries).
set(out "")
foreach(v F1 F2 F3 F4)
  if(DEFINED ${v})
    file(SHA256 "${${v}}" h)
    get_filename_component(n "${${v}}" NAME)
    string(APPEND out "${h} ${n}\n")
  endif()
endforeach()
file(WRITE "${OUT}" "${out}")
