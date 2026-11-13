// SPDX-License-Identifier: MIT
// A stand-in for another hook layer's module (tests/test_wrapped_list.cpp, case F): it holds the class vtables of that
// layer's command-list and device wrapper objects, as a game interface's interposer does. The test fills the tables.
// Written for this project.
extern "C" __declspec(dllexport) void** WgTestWrapperListTable() {
    static void* table[128];
    return table;
}

extern "C" __declspec(dllexport) void** WgTestWrapperDeviceTable() {
    static void* table[64];
    return table;
}
