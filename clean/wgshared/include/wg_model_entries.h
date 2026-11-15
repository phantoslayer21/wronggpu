// SPDX-License-Identifier: MIT
#pragma once
// The ONE definition of where WrongGPU's model entries are in the list of upscaler versions the front end gives a host
// (FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: frontend.cpp ListVersions, from the registry's SR model rows). A host such as
// OptiScaler stores the player's entry as its position in that list (UpscalerIndex; OptiScaler shows the list sorted, but
// the index it stores is the position in the list the DLL returned), so this order is what the launcher writes:
//   Model M first, then Model K (a build with Model K), then Model L (a build with Model L).
// The launcher (profile.h: kModelKEntryIndex / kModelLEntryIndex) and the front end use it; test_frontend
// (TestModelEntryOrder) fails when the list ListVersions returns and this order disagree, for each build's K / L flags.
// The index is for display only: which model runs is the model choice record's (clean/fsr/src/model_choice_record.h).

namespace wgentries {

// The position of `model`'s entry ('M', 'K', 'L') in a build with or without Model K and Model L; -1 when the build does
// not list that model.
constexpr int ModelEntryIndex(char model, bool withK, bool withL) {
    return model == 'M' ? 0 : model == 'K' ? (withK ? 1 : -1) : model == 'L' ? (withL ? (withK ? 2 : 1) : -1) : -1;
}

// The same as the decimal text an ini holds ("0", "1", "2"; nullptr for -1).
constexpr const char* IndexText(int index) { return index == 0 ? "0" : index == 1 ? "1" : index == 2 ? "2" : nullptr; }

}  // namespace wgentries
