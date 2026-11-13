// SPDX-License-Identifier: MIT
// D3D12 headers for the interop. MinGW's widl-generated headers return aggregates (descriptor handles, LUIDs) from
// COM methods by value, which does not match the Microsoft ABI unless WIDL_EXPLICIT_AGGREGATE_RETURNS is defined.
#pragma once

#if defined(__MINGW32__) && !defined(WIDL_EXPLICIT_AGGREGATE_RETURNS)
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
