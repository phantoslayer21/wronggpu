// SPDX-License-Identifier: MIT
#pragma once
// Original query continuation for D3D12 command-list splits. Based on Microsoft's
// public CountersAndQueries specification; no SDK/driver implementation copied.
#include <d3d12.h>
#include <vector>

namespace wgwin {
struct StatisticsQuery { ID3D12QueryHeap* heap; D3D12_QUERY_TYPE type; UINT index; };
using QueryBeginFn = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12QueryHeap*, D3D12_QUERY_TYPE, UINT);
using QueryEndFn = QueryBeginFn;
using QueryResolveFn = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12QueryHeap*, D3D12_QUERY_TYPE, UINT, UINT, ID3D12Resource*, UINT64);
struct QueryMethods { QueryBeginFn begin; QueryEndFn end; QueryResolveFn resolve; };
bool IsStatisticsQuery(D3D12_QUERY_TYPE type);
bool InternalQueryCommands();
struct InternalQueryScope { InternalQueryScope(); ~InternalQueryScope(); };
void StatisticsQueryBegin(const StatisticsQuery& query);
bool PrepareStatisticsQueries(const std::vector<StatisticsQuery>& queries);
void SuspendStatisticsQueries(ID3D12GraphicsCommandList* list, const std::vector<StatisticsQuery>& queries, QueryMethods methods);
void ResumeStatisticsQueries(ID3D12GraphicsCommandList* list, const std::vector<StatisticsQuery>& queries, QueryMethods methods);
bool HasStatisticsPrefix(ID3D12QueryHeap* heap, D3D12_QUERY_TYPE type, UINT first, UINT count);
void ResolveStatisticsQuery(ID3D12GraphicsCommandList* list, const StatisticsQuery& query, ID3D12Resource* target, UINT64 offset, QueryMethods methods);
}
