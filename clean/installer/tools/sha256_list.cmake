# SPDX-License-Identifier: MIT
# cmake -DF1=<file> [-DF2=<file> -DF3=<file>] -DOUT=<list> -P sha256_list.cmake: one line "<sha256> <file name>" per file
# (the testing build's list of WrongGPU.exe and the launcher suite's canaries, tests/launcher_build_facts.py
# verified_canaries).
set(out "")
foreach(v F1 F2 F3)
  if(DEFINED ${v})
    file(SHA256 "${${v}}" h)
    get_filename_component(n "${${v}}" NAME)
    string(APPEND out "${h} ${n}\n")
  endif()
endforeach()
file(WRITE "${OUT}" "${out}")
